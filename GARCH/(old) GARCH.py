import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy.stats import norm, t, gennorm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython import get_ipython

def generate_resid(variable: str, start_date: str, end_date: str, transformed_data: pd.DataFrame):
    """
    Generate residuals by dynamically retrieving train and test predictions.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Copy transformed data and assign correct date index
    transformed_data_copy = transformed_data.copy()
    transformed_data_copy.index = date_range

    # Retrieve notebook variables dynamically
    ipython = get_ipython()
    train_predict = ipython.user_ns.get(f"{variable.lower()}_train_predict", None)
    test_predict = ipython.user_ns.get(f"{variable.lower()}_test_predict", None)

    if train_predict is None or test_predict is None:
        raise NameError(f"train_predict or test_predict not found for {variable}. Ensure they are defined.")

    # Convert columns into time series
    transformed_series = pd.Series(data=transformed_data_copy[variable], index=date_range, name=f"{variable}")

    # Create an empty time series for the full date range
    merged_predictions_series = pd.Series(index=date_range, dtype=float, name=f"merged_predictions_series_{variable.lower()}")

    # Fill in training and test predictions dynamically
    merged_predictions_series.update(train_predict)
    merged_predictions_series.update(test_predict)

    # Compute residuals
    resid_complete = transformed_series - merged_predictions_series
    resid_complete.name = f"{variable.lower()}_resid_complete"
    display(resid_complete)

    # Get training residuals dynamically
    resid_train = ipython.user_ns.get(f"{variable.lower()}_model_results", None)

    if resid_train is None:
        raise NameError(f"Model results not found for {variable}. Ensure the model was trained correctly.")

    resid_train = resid_train.resid
    resid_train.name = f"{variable.lower()}_resid_train"
    display(resid_train)

    return resid_complete, resid_train
    
def generate_resid_append(variable: str, start_date: str, end_date: str, transformed_data: pd.DataFrame):
    """
    Generate residuals by dynamically retrieving train and test predictions.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Copy transformed data and assign correct date index
    transformed_data_copy = transformed_data.copy()
    transformed_data_copy.index = date_range

    # Retrieve notebook variables dynamically
    ipython = get_ipython()
    train_predict = ipython.user_ns.get(f"{variable.lower()}_train_predict", None)
    test_predict = ipython.user_ns.get(f"{variable.lower()}_test_predict", None)

    if train_predict is None or test_predict is None:
        raise NameError(f"train_predict or test_predict not found for {variable}. Ensure they are defined.")

    # Convert columns into time series
    transformed_series = pd.Series(data=transformed_data_copy[variable], index=date_range, name=f"{variable}")

    # Create an empty time series for the full date range
    merged_predictions_series = pd.Series(index=date_range, dtype=float, name=f"merged_predictions_series_{variable.lower()}")

    # Fill in training and test predictions dynamically
    merged_predictions_series.update(train_predict)
    merged_predictions_series.update(test_predict)

    # Compute residuals using the same source for training
    resid_train = ipython.user_ns.get(f"{variable.lower()}_model_results", None)

    if resid_train is None:
        raise NameError(f"Model results not found for {variable}. Ensure the model was trained correctly.")

    resid_train = resid_train.resid
    resid_train.name = f"{variable.lower()}_resid_train"
    
    # Use resid_train instead of train_predict to ensure consistency
    merged_predictions_series.loc[:resid_train.index[-1]] = transformed_series.loc[:resid_train.index[-1]] - resid_train

    resid_complete = transformed_series - merged_predictions_series
    resid_complete.name = f"{variable.lower()}_resid_complete"
    
    display(resid_complete)
    display(resid_train)

    return resid_complete, resid_train

def garch_testing(residuals, variable: str, max_lag=20):
    """
    Perform ARCH LM test and Portmanteau-Q (Ljung-Box) test for squared residuals,
    and generate diagnostic plots.
    """
    # Plot residuals and squared residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals, label="Residuals", color="blue")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"SARIMAX {variable} Residuals")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(np.square(residuals), label="Squared Residuals", color="green")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"Squared Residuals (SARIMAX {variable})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot ACF and PACF of squared residuals
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(np.square(residuals), lags=max_lag, ax=axes[0])
    axes[0].set_title(f"ACF of Squared Residuals ({variable})")
    plot_pacf(np.square(residuals), lags=max_lag, method="ywm", ax=axes[1])
    axes[1].set_title(f"PACF of Squared Residuals ({variable})")
    plt.tight_layout()
    plt.show()
    
    # Perform McLeod-Li Test (Ljung-Box on squared residuals)
    mcleod_li_test = acorr_ljungbox(np.square(residuals), lags=max_lag, return_df=True)
    plt.figure(figsize=(10, 6))
    plt.plot(mcleod_li_test.index, mcleod_li_test['lb_pvalue'], 'o-', label='P-values')
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Significance Threshold (0.05)')
    plt.title(f"McLeod-Li Test P-Values ({variable})")
    plt.xlabel("Lag")
    plt.ylabel("P-value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Compute ARCH LM and Ljung-Box tests
    lags = list(range(1, max_lag + 1))
    arch_lm_stats, arch_lm_pvalues = [], []
    for lag in lags:
        arch_test = het_arch(residuals, nlags=lag)
        arch_lm_stats.append(arch_test[0])
        arch_lm_pvalues.append(arch_test[1])
    ljungbox_results = acorr_ljungbox(np.square(residuals), lags=lags, return_df=True)
    
    # Store results in DataFrame
    output_df = pd.DataFrame({
        "Lag": lags,
        "PQ Test Stat": ljungbox_results['lb_stat'].values,
        "PQ p-value": ljungbox_results['lb_pvalue'].values,
        "LM Test Stat": arch_lm_stats,
        "LM p-value": arch_lm_pvalues,
    })
    
    return output_df

def evaluate_garch(data, max_order=5, distributions=["normal", "t", "ged"]):
    """
    Evaluate zero-mean GARCH models with different parameter orders and error term distributions.

    Parameters:
        data (array-like): Time series data for fitting the GARCH model.
        max_order (int): Maximum order for p and q in the GARCH(p, q) model.
        distributions (list): List of error term distributions to test.

    Returns:
        pd.DataFrame: A table showing the AIC, BIC, and residual ARCH test results.
    """
    import pandas as pd
    import numpy as np
    from arch import arch_model
    from statsmodels.stats.diagnostic import het_arch

    # Ensure data is a Pandas Series or a 1D NumPy array
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise ValueError("Input data must be a Pandas Series or a 1D NumPy array.")

    # Drop missing values and scale data if needed
    data = pd.Series(data).dropna()  # Convert to Series and drop NaNs
    if data.abs().mean() < 1e-3:
        data = data * 100  # Rescale data to improve numerical stability

    results = []

    # Iterate through GARCH(p, q) orders
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            # Skip invalid combinations where both p and q are 0
            if p == 0 and q == 0:
                continue
            print(p, q)
            for dist in distributions:
                try:
                    # Fit GARCH model with Zero mean
                    model = arch_model(data, vol="Garch", p=p, q=q, dist=dist, mean="Zero")
                    result = model.fit(disp="off", options={"maxiter": 1000})

                    # Extract information criteria
                    aic = result.aic
                    bic = result.bic

                    # Perform ARCH test on residuals
                    arch_test = het_arch(result.resid)
                    arch_stat = arch_test[0]
                    arch_pval = arch_test[1]

                    # Append results
                    results.append({
                        "p": p,
                        "q": q,
                        "Distribution": dist,
                        "AIC": aic,
                        "BIC": bic,
                        "ARCH Stat": arch_stat,
                        "ARCH p-value": arch_pval
                    })
                except Exception as e:
                    # Print detailed error for debugging
                    print(f"Error for GARCH({p},{q}) with {dist} distribution: {e}")
                    results.append({
                        "p": p,
                        "q": q,
                        "Distribution": dist,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "ARCH Stat": np.nan,
                        "ARCH p-value": np.nan
                    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort results by AIC (default criterion)
    results_df = results_df.sort_values(by="AIC").reset_index(drop=True)

    return results_df

def fit_garch(variable: str, vol: str, p: int, q: int, dist: str, last_obs: str):
    """
    Fit a GARCH(p, q) model to the residuals.
    """
    ipython = get_ipython()
    residuals = ipython.user_ns.get(f"{variable.lower()}_resid_complete", None)

    # Check last few values before fitting
    print("Last values in resid_complete before fitting:")
    print(residuals.loc[:last_obs].tail(10))

    garch_model = arch_model(residuals, vol=vol, p=p, q=q, dist=dist, mean="Zero")
    garch_fitted = garch_model.fit(last_obs=last_obs, disp="off")
    print(garch_fitted.summary())
    return garch_fitted

def fit_garch_verify(variable: str, vol: str, p: int, q: int, dist: str, last_obs: str):
    """
    Fit a GARCH(p, q) model to the residuals.
    """
    ipython = get_ipython()
    residuals = ipython.user_ns.get(f"{variable.lower()}_resid_train", None)

    # Check last few values before fitting
    print("Last values in resid_train before fitting:")
    print(residuals.loc[:last_obs].tail(10))

    garch_model = arch_model(residuals, vol=vol, p=p, q=q, dist=dist, mean="Zero")
    garch_fitted = garch_model.fit(last_obs=last_obs, disp="off")
    print(garch_fitted.summary())
    return garch_fitted

def evaluate_egarch(data, max_order=5, distributions=["normal", "t", "ged"]):
    """
    Evaluate EGARCH models with different parameter orders and error term distributions.

    Parameters:
        data (array-like): Time series data for fitting the EGARCH model.
        max_order (int): Maximum order for p and q in the EGARCH(p, q) model.
        distributions (list): List of error term distributions to test.

    Returns:
        pd.DataFrame: A table showing the AIC, BIC, and residual ARCH test results.
    """
    import pandas as pd
    import numpy as np
    from arch import arch_model
    from statsmodels.stats.diagnostic import het_arch

    # Ensure data is a Pandas Series or a 1D NumPy array
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise ValueError("Input data must be a Pandas Series or a 1D NumPy array.")

    # Drop missing values and scale data if needed
    data = pd.Series(data).dropna()  # Convert to Series and drop NaNs
    if data.abs().mean() < 1e-3:
        data = data * 100  # Rescale data to improve numerical stability

    results = []

    # Iterate through EGARCH(p, q) orders
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            # Skip invalid combinations where both p and q are 0
            if p == 0 and q == 0:
                continue

            for dist in distributions:
                try:
                    # Fit EGARCH model
                    model = arch_model(data, vol="EGARCH", p=p, q=q, dist=dist, mean="Zero")
                    result = model.fit(disp="off", options={"maxiter": 1000})

                    # Extract information criteria
                    aic = result.aic
                    bic = result.bic

                    # Perform ARCH test on residuals
                    arch_test = het_arch(result.resid)
                    arch_stat = arch_test[0]
                    arch_pval = arch_test[1]

                    # Append results
                    results.append({
                        "p": p,
                        "q": q,
                        "Distribution": dist,
                        "AIC": aic,
                        "BIC": bic,
                        "ARCH Stat": arch_stat,
                        "ARCH p-value": arch_pval
                    })
                except Exception as e:
                    # Print detailed error for debugging
                    print(f"Error for EGARCH({p},{q}) with {dist} distribution: {e}")
                    results.append({
                        "p": p,
                        "q": q,
                        "Distribution": dist,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "ARCH Stat": np.nan,
                        "ARCH p-value": np.nan
                    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort results by AIC (default criterion)
    results_df = results_df.sort_values(by="AIC").reset_index(drop=True)

    return results_df

def garch_predict(variable: str, dist: str, model: str, start_date: str, end_date: str):
    """
    Generate predictions using a SARIMAX-GARCH or SARIMAX-EGARCH model.
    """
    
    rs = np.random.RandomState(42)

    #Retrieve global variables dynamically
    ipython = get_ipython()
    diff_exog_test = ipython.user_ns.get("diff_exog_test", None)
    test_predict = ipython.user_ns.get(f"appended_{variable.lower()}", None).predict(
        start=start_date, end=end_date, exog=diff_exog_test, dynamic=True
    )
    forecast_horizon = len(test_predict)

    garch_model = ipython.user_ns.get(f"{variable.lower()}_{model.lower()}_fitted", None)
    garch_forecast = garch_model.forecast(
        horizon=forecast_horizon, start=start_date, align='origin', reindex=False, method='simulation', simulations=100
    )
    
    forecasted_variance = garch_forecast.variance.iloc[-1, :forecast_horizon].values
    forecasted_stddev = np.sqrt(forecasted_variance)
    
    if dist == "normal":
        simulated_z = rs.normal(loc=0, scale=1, size=forecast_horizon)
    elif dist == "t":
        simulated_z = rs.standard_t(df=garch_model.params['nu'], size=forecast_horizon)
    elif dist == "ged":
        simulated_z = gennorm.rvs(beta=garch_model.params['nu'], size=forecast_horizon, random_state=rs)
    else:
        raise ValueError("Invalid distribution. Choose from 'normal', 't', or 'ged'.")
    
    predicted_et = forecasted_stddev * simulated_z
    predicted_et = pd.Series(predicted_et, index=test_predict.index)
    combined_predictions = test_predict + predicted_et
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ipython.user_ns.get(f"{variable.upper()}_train").plot(ax=ax, label='Train Data', color='blue')
    test_predict.plot(ax=ax, label='SARIMAX Predictions (mu)', color='orange')
    ipython.user_ns.get(f"{variable.lower()}_train_predict").plot(ax=ax, label='Train Set Predictions', color='green')
    combined_predictions.plot(ax=ax, label=f'SARIMAX-{model.upper()} Predictions', color='red')
    ax.set_title(f'Predictions with SARIMAX-{model.upper()} Model ({variable.upper()})')
    ax.legend()
    plt.show()
    
    return combined_predictions

def metrics(test_actual: pd.Series, test_predictions: pd.Series, variable: str, model: str):
    """
    Compute MAE and RMSE for given test actual and predicted values.
    """
    mae_test = mean_absolute_error(test_actual, test_predictions)
    mse_test = mean_squared_error(test_actual, test_predictions)
    rmse_test = np.sqrt(mse_test)
    
    print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
    print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse_test}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    test_actual.plot(ax=ax, label='Original Data')
    test_predictions.plot(ax=ax, label='Predictions on Test Set', linestyle='--')
    ax.set_title(f'SARIMAX-{model.upper()} {variable.upper()} Predictions')
    ax.legend()
    plt.show()
    
    return mae_test, rmse_test
