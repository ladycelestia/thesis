import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random
import joblib

from transform import classify_features
from datetime import datetime, timedelta


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')


import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True

import ray
from ray import tune, train
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from ray.train import Checkpoint, get_checkpoint, RunConfig, report
from ray.tune.stopper import TrialPlateauStopper

from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, TransformerMixin

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):       
        return torch.sqrt(self.mse(predictions, targets)+1e-6)
    
class TimeSeriesDataset(Dataset):
    
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len 
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

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


    
def train_model(model, dataloader, device, optimizer, lambda_reg, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        l1_norm = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
        loss += lambda_reg * l1_norm
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    
    model.eval()
    total_loss = 0.0  # Initialize total loss
    total_samples = 0  # Initialize total sample count to 0

    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.to(device)

        # Forward pass
        outputs = model(inputs)

        predictions = outputs.to(device)

        loss = criterion(predictions, target)

        # Accumulate the loss and sample count
        total_loss += loss.item() * inputs.size(0)  # Total loss for the batch
        total_samples += inputs.size(0)  # Number of samples in the current batch


    return total_loss / total_samples if total_samples > 0 else 0.0  # Avoid division by zero

def load_data():
    train_dataset = TimeSeriesDataset(train_data, train_labels, seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = TimeSeriesDataset(val_data, val_labels, seq_len)    
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False) 
    
    return train_dataloader, val_dataloader

def raytrain(config,epoch):
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
    lambda_reg=config["lambda_reg"]
    norm_layer_type = config["norm_layer_type"]
    activation_fn = activation_fn_mapping[config["activation_fn"]]
    activation_fn1 = activation_fn_mapping[config["activation_fn1"]]
    model = LSTMModel(input_size, config["hidden_size"], output_size, config["num_layers"],config["dropout"],activation_fn=activation_fn,activation_fn1=activation_fn1, norm_layer_type=norm_layer_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  config["factor"],config["patience"])
    
    train_dataset = TimeSeriesDataset(train_data, train_labels, seq_len=config["seq_len"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_dataset = TimeSeriesDataset(val_data, val_labels, seq_len=config["seq_len"])    
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False) 
    for e in range(epoch):  # Replace with your actual number of epochs
        train_model(model, train_dataloader, device, optimizer, lambda_reg,train_criterion)
        val_loss = evaluate(model, val_dataloader, device, test_criterion)
        scheduler.step(val_loss)

        train.report(
            {
                "loss": val_loss
            }  
        )

@torch.no_grad()
def test_best_model(optimized_model,batch_size,seq_len):
    
    test_loader = load_test(batch_size,seq_len) 
    optimized_model.eval()

    wap_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            outputs = optimized_model(inputs)
            predictions = outputs.to(device)
            
            all_predictions.append(predictions)
            all_targets.append(target)

    # Concatenate all predictions and targets for inverse transformation
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()  
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy() 
    
    # Apply inverse transformations to predictions and targets
    all_predictions_2d = np.zeros((all_predictions.shape[0], 2))  
    #all_predictions_2d[:, 1] = all_predictions[:, 0]  # Fill only the first column with predictions (LWAP)
    all_predictions_2d[:, 0] = all_predictions[:, 0]  # Fill only the first column with predictions (GWAP)
    
    all_predictions_inverse = boxcoxy_fit_loaded.inverse_transform(all_predictions_2d)[:, 0] # GWAP
    #all_predictions_inverse = boxcoxy_fit_loaded.inverse_transform(all_predictions_2d)[:, 1] # LWAP

  
    predictions_output = all_predictions_inverse
    targets_output = all_targets.squeeze()

    
    wap_untransformed_loss = test_criterion(
        torch.tensor(predictions_output, dtype=torch.float32).to(device),
        torch.tensor(targets_output, dtype=torch.float32).to(device)
    )

    # Accumulate the losses
    wap_loss += wap_untransformed_loss.item() * predictions_output.shape[0]
    total_samples += predictions_output.shape[0]
  
    # Calculate and return the average loss
    average_wap_loss = wap_loss / total_samples if total_samples > 0 else 0.0
    print("Average GWAP Loss:", average_wap_loss)

    return average_wap_loss, all_predictions_inverse

def plot_predictions(all_predictions_inverse,island):
    
    output_folder =  os.path.join(r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\DAILY', island)
    
    actual_gwap = np.array(col['GWAP'])  # Convert to NumPy array
    #actual_lwap = np.array(col['LWAP'])  # Convert to NumPy array

    
    if(island=="Mindanao"):
        full_date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D").to_numpy()
    else:
        full_date_range = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D").to_numpy()

    
    test_data_points = all_predictions_inverse.shape[0]
    test_start_date = datetime(2023, 12, 31) - timedelta(days=test_data_points - 1)
    test_date_range = pd.date_range(start=test_start_date, periods=test_data_points, freq="D").to_numpy()


    
    wap_predictions = all_predictions_inverse
    

    # Plotting

    plt.figure(figsize=(12, 6))

    plt.plot(full_date_range, actual_gwap, label='GWAP Actual (Full Data)', color='blue')
    #plt.plot(full_date_range, actual_lwap, label='LWAP Actual (Full Data)', color='green')
    plt.plot(test_date_range, wap_predictions, label='GWAP Prediction', color='red')

    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{island} GWAP Predictions vs Actual Values')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{island}_preds.png'))
    plt.show()

def main(epoch,island,trials):
    set_seed(1)
    search_space = {
        "hidden_size": tune.choice([16,32, 64, 128, 256, 512]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
        "activation_fn": tune.choice(["tanh","relu"]),
        "activation_fn1": tune.choice(["sigmoid","identity"]),
        "seed": 1,
        "num_layers": tune.choice([1,2, 3,4]),
        "patience": tune.choice([5, 10, 20]),
        "factor": tune.uniform(0.1, 0.5),
        "seq_len":tune.choice([3,5,7,9]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "norm_layer_type": tune.choice(['batch_norm','layer_norm','none']),
        "lambda_reg": tune.loguniform(1e-5, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=50,
        reduction_factor=2
    )
    plateau_stopper = TrialPlateauStopper(metric="val_loss",mode="min")

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"

    trainable_with_params = tune.with_parameters(raytrain, epoch=epoch)


    trainable_with_resources = tune.with_resources(trainable_with_params, resources={"cpu": 4, "gpu": 0.2})

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

    activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }
    optimized_lambda_reg = best_result.config["lambda_reg"]
    optimized_activation_fn = activation_fn_mapping.get(best_result.config["activation_fn"])
    optimzed_activation_fn1 = activation_fn_mapping.get(best_result.config["activation_fn1"])
    optimized_norm_layer_type = best_result.config["norm_layer_type"]
    
    # Step 2: Training and testing with different seeds
    wap_loss = 0.0

    num_seeds = 10  # Total seeds to test
    train_dataset = TimeSeriesDataset(train_data, train_labels, best_result.config["seq_len"])
    train_dataloader = DataLoader(train_dataset, best_result.config["batch_size"], shuffle=False)

    val_dataset = TimeSeriesDataset(val_data, val_labels, best_result.config["seq_len"])    
    val_dataloader = DataLoader(val_dataset, best_result.config["batch_size"], shuffle=False) 
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
            activation_fn1=optimzed_activation_fn1,
            norm_layer_type=optimized_norm_layer_type
        ).to(device)

        optimizer_op = optim.AdamW(
            optimized_model.parameters(),
            lr=best_result.config["lr"],
            weight_decay=best_result.config["weight_decay"]
        )
        scheduler_op = ReduceLROnPlateau(optimizer_op, 'min', factor=best_result.config["factor"], patience=best_result.config["patience"])

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
        wap_seed_loss, pred_inverse = test_best_model(optimized_model,best_result.config["batch_size"],best_result.config["seq_len"]) 
        wap_loss += wap_seed_loss

        all_pred_inverses.append(np.array(pred_inverse))

# At the end, you’ll have accumulated separate gwap and lwap losses across all seeds
    average_wap_loss = wap_loss / num_seeds
    print("Average GWAP test loss over {} runs: {:.4f}".format(num_seeds, average_wap_loss))

    stacked_predictions = np.stack(all_pred_inverses)
    avg_pred_inverse = np.mean(stacked_predictions, axis=0)  

# Plot the averaged predictions
    output_folder =  os.path.join(r'D:/School\ADMU\4Y\SEM 1\MATH 199.11\Final\DAILY', island)
    plot_predictions(avg_pred_inverse,island)

    for i in range(num_seeds):
        plt.figure(figsize=(10, 5))
        plt.plot(all_train_losses[i], label=f'Training Loss Seed {i+1}')
        plt.plot(all_val_losses[i], label=f'Validation Loss Seed {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 0.5)
        plt.legend()
        plt.title(f'{island} Training and Validation Losses per Epoch for Seed {i+1}')
        plt.savefig(os.path.join(output_folder, f'{island} training_validation_loss_seed_{i+1}.png'))
        plt.show()
    return best_result.config, average_wap_loss
