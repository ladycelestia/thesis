import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

import ray
from ray import tune, train
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import TrialPlateauStopper
from ray.train import RunConfig

import transformscript, datascript

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom loss function
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):       
        return torch.sqrt(self.mse(predictions, targets) + 1e-6)

class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / (target + self.epsilon)))
L1 = nn.L1Loss()
mape_fn = MAPELoss()
# Dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len 
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

# Set random seed for reproducibility
def set_seed(seed=None):
    if seed is None:
        seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

# Training function
def train_model(model, dataloader, device, optimizer, lambda_reg, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)


        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))

        
        # L1 regularization
        l1_norm = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
        loss += lambda_reg * l1_norm
        
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

# Evaluation function
@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

# Ray Tune training function
def raytrain(config, epoch, train_data, val_data, train_labels, val_labels, input_size, output_size, train_criterion, test_criterion, LSTMModel):
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    set_seed(config["seed"])
    
    activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }
    
    lambda_reg = config["lambda_reg"]
    norm_layer_type = config["norm_layer_type"]
    activation_fn = activation_fn_mapping[config["activation_fn"]]
    activation_fn1 = activation_fn_mapping[config["activation_fn1"]]
    
    model = LSTMModel(
        input_size, 
        config["hidden_size"], 
        output_size, 
        config["num_layers"],
        config["dropout"],
        activation_fn=activation_fn,
        activation_fn1=activation_fn1, 
        norm_layer_type=norm_layer_type
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', config["factor"], config["patience"])
    
    train_dataset = TimeSeriesDataset(train_data, train_labels, seq_len=config["seq_len"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_dataset = TimeSeriesDataset(val_data, val_labels, seq_len=config["seq_len"])    
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False) 
    
    for e in range(epoch):
        train_model(model, train_dataloader, device, optimizer, lambda_reg, train_criterion)
        val_loss = evaluate(model, val_dataloader, device, test_criterion)
        scheduler.step(val_loss)

        train.report({"loss": val_loss})

# Test function
@torch.no_grad()
def test_best_model(optimized_model, test_loader, test_criterion, regionname, target_label, device):
    optimized_model.eval()

    wap_loss = 0.0
    mae_loss = 0.0
    mape_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []


    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)         
        outputs = optimized_model(inputs)
        all_predictions.append(outputs)
        all_targets.append(targets)

    # Concatenate all predictions and targets for inverse transformation
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()  
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy() 

    # Inverse transform
    all_predictions_inverse = transformscript.inverse_transform_data(all_predictions, regionname, target_label)
  
    predictions_output = all_predictions_inverse
    targets_output = all_targets

    # Compute loss
    predictions_tensor = torch.tensor(predictions_output, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets_output, dtype=torch.float32, device=device)
    
    wap_untransformed_loss = test_criterion(predictions_tensor, targets_tensor)

    # Accumulate the losses
    wap_loss += wap_untransformed_loss.item() * predictions_output.shape[0]
    total_samples += predictions_output.shape[0]

    mae_batch_loss = L1(predictions_tensor, targets_tensor).item()
    mape_batch_loss = mape_fn(predictions_tensor, targets_tensor).item()

    # Accumulate losses
    mae_loss += mae_batch_loss * predictions_tensor.shape[0]
    mape_loss += mape_batch_loss * predictions_tensor.shape[0]

    # Calculate and return the average loss
    average_wap_loss = wap_loss / total_samples if total_samples > 0 else 0.0
    average_mae_loss = mae_loss / total_samples if total_samples > 0 else 0.0
    average_mape_loss = mape_loss / total_samples if total_samples > 0 else 0.0


    return average_wap_loss, average_mae_loss, average_mape_loss, all_predictions_inverse

# Function to plot predictions
def plot_predictions(all_predictions_inverse, actual_wap, target_label,island, output_folder):
    
    if island == "Mindanao":
        full_date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D").to_numpy()
    else:
        full_date_range = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D").to_numpy()
    
    test_data_points = all_predictions_inverse.shape[0]
    test_start_date = datetime(2023, 12, 31) - timedelta(days=test_data_points - 1)
    test_date_range = pd.date_range(start=test_start_date, periods=test_data_points, freq="D").to_numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.plot(full_date_range, actual_wap.values, label=f'{target_label} Actual (Full Data)', color='blue')
    plt.plot(test_date_range, all_predictions_inverse, label=f'{target_label} Prediction', color='red')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{island} {target_label} Predictions vs Actual Values')
    plt.legend()
    
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{island}_{target_label}preds.png'))
    plt.show()

def load_test_loader(test_data, regionname, label_name, batch_size, seq_len):
    _, _, u_test_labels= datascript.split_data(datascript.load_data(regionname,label_name,False,False))
    u_test_labels = u_test_labels.values
    test_dataset = TimeSeriesDataset(test_data, u_test_labels, seq_len)
    return DataLoader(test_dataset, batch_size, shuffle=False)

# Main tuning and training function
def tune_and_train(train_data, val_data, test_data, train_labels, val_labels, 
                   input_size, output_size, epoch, island, trials, LSTMModel, 
                   train_criterion, test_criterion, actual_wap, regionname, target_label,output_base_folder):
    set_seed(1)
    
    output_folder = os.path.join(output_base_folder, island)
    os.makedirs(output_folder, exist_ok=True)
    
    search_space = {
        "hidden_size": tune.choice([16, 32, 64, 128, 256, 512]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
        "activation_fn": tune.choice(["tanh", "relu"]),
        "activation_fn1": tune.choice(["sigmoid", "identity"]),
        "seed": 1,
        "num_layers": tune.choice([1, 2, 3, 4]),
        "patience": tune.choice([5, 10, 20]),
        "factor": tune.uniform(0.1, 0.5),
        "seq_len": tune.choice([3, 5, 7, 9]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "norm_layer_type": tune.choice(['batch_norm', 'layer_norm', 'none']),
        "lambda_reg": tune.loguniform(1e-5, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=50,
        reduction_factor=2
    )
    
    plateau_stopper = TrialPlateauStopper(metric="val_loss", mode="min")

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"

    trainable_with_params = tune.with_parameters(
        raytrain, 
        epoch=epoch,
        train_data=train_data,
        val_data=val_data,
        train_labels=train_labels,
        val_labels=val_labels,
        input_size=input_size,
        output_size=output_size,
        train_criterion=train_criterion,
        test_criterion=test_criterion,
        LSTMModel=LSTMModel
    )

    trainable_with_resources = tune.with_resources(
        trainable_with_params, 
        resources={"cpu": 4, "gpu": 0.2}
    )

    # Step 1: Hyperparameter tuning
    tuner = Tuner(
        trainable_with_resources,  
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric="loss", mode="min"),
            num_samples=trials,
            scheduler=scheduler,
            trial_dirname_creator=trial_dirname_creator
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            verbose=1,
            stop=plateau_stopper
        )
    )
    
    # Run the tuner and collect the results
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))

    # Get optimized parameters
    activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }
    
    optimized_lambda_reg = best_result.config["lambda_reg"]
    optimized_activation_fn = activation_fn_mapping.get(best_result.config["activation_fn"])
    optimized_activation_fn1 = activation_fn_mapping.get(best_result.config["activation_fn1"])
    optimized_norm_layer_type = best_result.config["norm_layer_type"]
    
    # Step 2: Training and testing with different seeds
    wap_loss = 0.0
    mae_loss=0.0
    mape_loss=0.0
    num_seeds = 10  # Total seeds to test
    
    train_dataset = TimeSeriesDataset(train_data, train_labels, best_result.config["seq_len"])
    train_dataloader = DataLoader(train_dataset, best_result.config["batch_size"], shuffle=False)

    val_dataset = TimeSeriesDataset(val_data, val_labels, best_result.config["seq_len"])    
    val_dataloader = DataLoader(val_dataset, best_result.config["batch_size"], shuffle=False)
    
    test_loader = load_test_loader(test_data, regionname, target_label, best_result.config["batch_size"], best_result.config["seq_len"])
    
    all_pred_inverses = []
    all_train_losses = []
    all_val_losses = []
    
    for seed in range(1, num_seeds + 1):
        set_seed(seed)
        optimized_model = LSTMModel(
            input_size=input_size,
            hidden_size=best_result.config["hidden_size"],
            output_size=output_size,
            num_layers=best_result.config["num_layers"],
            dropout=best_result.config["dropout"],
            activation_fn=optimized_activation_fn,
            activation_fn1=optimized_activation_fn1,
            norm_layer_type=optimized_norm_layer_type
        ).to(device)

        optimizer_op = optim.AdamW(
            optimized_model.parameters(),
            lr=best_result.config["lr"],
            weight_decay=best_result.config["weight_decay"]
        )
        
        scheduler_op = ReduceLROnPlateau(
            optimizer_op, 
            'min', 
            factor=best_result.config["factor"], 
            patience=best_result.config["patience"]
        )

        train_losses = []
        val_losses = []

        # Train the model with the best configuration and the current seed
        for e in range(epoch):
            train_loss = train_model(optimized_model, train_dataloader, device, optimizer_op, optimized_lambda_reg, train_criterion)
            val_loss = evaluate(optimized_model, val_dataloader, device, test_criterion)
            scheduler_op.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Test the model using the trained optimized_model
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        wap_seed_loss, mae_seed_loss,mape_seed_loss, pred_inverse = test_best_model(optimized_model, test_loader, test_criterion, regionname, target_label,device) 
        wap_loss += wap_seed_loss
        mae_loss += mae_seed_loss
        mape_loss += mape_seed_loss
        all_pred_inverses.append(np.array(pred_inverse))

    # Calculate average loss and predictions
    average_wap_loss = wap_loss / num_seeds
    average_mae_loss = mae_loss / num_seeds
    average_mape_loss = mape_loss / num_seeds

    print(f"Average {target_label} RMSE loss over {num_seeds} runs: {average_wap_loss:.4f}")
    print(f"Average {target_label} MAE loss over {num_seeds} runs: {average_mae_loss:.4f}")
    print(f"Average {target_label} MAPE loss over {num_seeds} runs: {average_mape_loss:.4f}")


    stacked_predictions = np.stack(all_pred_inverses)
    avg_pred_inverse = np.mean(stacked_predictions, axis=0)  

    # Plot the averaged predictions
    plot_predictions(avg_pred_inverse, actual_wap,target_label, island, output_folder)

    # Plot training/validation losses for each seed
    for i in range(num_seeds):
        plt.figure(figsize=(10, 5))
        plt.plot(all_train_losses[i], label=f'Training Loss Seed {i+1}')
        plt.plot(all_val_losses[i], label=f'Validation Loss Seed {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 0.5)
        plt.legend()
        plt.title(f'{island} {target_label} Training and Validation Losses per Epoch for Seed {i+1}')
        plt.savefig(os.path.join(output_folder, f'{island}_{target_label}_training_validation_loss_seed_{i+1}.png'))
        plt.show()
        
    return best_result.config, average_wap_loss,average_mae_loss,average_mape_loss, avg_pred_inverse

def run_hyperparams(config, train_data, val_data, test_data, train_labels, val_labels, 
                               input_size, output_size, epoch, island, LSTMModel, 
                               train_criterion, test_criterion, actual_wap, regionname, target_label, output_base_folder):
    num_seeds=10
    set_seed(1)
    
    output_folder = os.path.join(output_base_folder, island)
    os.makedirs(output_folder, exist_ok=True)
    
    activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }
    
    lambda_reg = config["lambda_reg"]
    activation_fn = activation_fn_mapping.get(config["activation_fn"])
    activation_fn1 = activation_fn_mapping.get(config["activation_fn1"])
    norm_layer_type = config["norm_layer_type"]
    
    # Step 2: Training and testing with different seeds
    wap_loss = 0.0
    mae_loss=0.0
    mape_loss=0.0
    
    train_dataset = TimeSeriesDataset(train_data, train_labels, config["seq_len"])
    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=False)

    val_dataset = TimeSeriesDataset(val_data, val_labels, config["seq_len"])    
    val_dataloader = DataLoader(val_dataset, config["batch_size"], shuffle=False)
    
    test_loader = load_test_loader(test_data, regionname, target_label, config["batch_size"], config["seq_len"])
    
    all_pred_inverses = []
    all_train_losses = []
    all_val_losses = []
    
    for seed in range(1, num_seeds + 1):
        set_seed(seed)
        optimized_model = LSTMModel(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            output_size=output_size,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            activation_fn=activation_fn,
            activation_fn1=activation_fn1,
            norm_layer_type=norm_layer_type
        ).to(device)

        optimizer = optim.AdamW(
            optimized_model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            'min', 
            factor=config["factor"], 
            patience=config["patience"]
        )

        train_losses = []
        val_losses = []

        # Train the model with the specified configuration and the current seed
        for e in range(epoch):
            train_loss = train_model(optimized_model, train_dataloader, device, optimizer, lambda_reg, train_criterion)
            val_loss = evaluate(optimized_model, val_dataloader, device, test_criterion)
            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Test the model
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        wap_seed_loss, mae_seed_loss,mape_seed_loss, pred_inverse = test_best_model(optimized_model, test_loader, test_criterion, regionname, target_label,device) 
        wap_loss += wap_seed_loss
        mae_loss += mae_seed_loss
        mape_loss += mape_seed_loss
        all_pred_inverses.append(np.array(pred_inverse))

    # Calculate average loss and predictions
    average_wap_loss = wap_loss / num_seeds
    average_mae_loss = mae_loss / num_seeds
    average_mape_loss = mape_loss / num_seeds

    print(f"Average {target_label} RMSE loss over {num_seeds} runs: {average_wap_loss:.4f}")
    print(f"Average {target_label} MAE loss over {num_seeds} runs: {average_mae_loss:.4f}")
    print(f"Average {target_label} MAPE loss over {num_seeds} runs: {average_mape_loss:.4f}")

    stacked_predictions = np.stack(all_pred_inverses)
    avg_pred_inverse = np.mean(stacked_predictions, axis=0)  

    # Plot the averaged predictions
    plot_predictions(avg_pred_inverse, actual_wap, target_label, island, output_folder)

    # Plot training/validation losses for each seed
    for i in range(num_seeds):
        plt.figure(figsize=(10, 5))
        plt.plot(all_train_losses[i], label=f'Training Loss Seed {i+1}')
        plt.plot(all_val_losses[i], label=f'Validation Loss Seed {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 0.5)
        plt.legend()
        plt.title(f'{island} {target_label} Training and Validation Losses per Epoch for Seed {i+1}')
        plt.savefig(os.path.join(output_folder, f'{island}_{target_label}_training_validation_loss_seed_{i+1}.png'))
        plt.show()
        
    return average_wap_loss, average_mae_loss,average_mape_loss, avg_pred_inverse

