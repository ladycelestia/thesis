import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy.stats import norm, t, gennorm
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_resid(
    variable: str,
    start_date: str,
    end_date: str,
    train,
    test,
    train_predict,
    test_predict,
    model_results
):
    """
    Generate residuals by dynamically retrieving train and test predictions.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create transformed series
    transformed_series = pd.Series(index=date_range, dtype=float, name=f"{variable.lower()}_complete")
    transformed_series.update(train)
    transformed_series.update(test)
    
    # Create prediction series
    merged_predictions_series = pd.Series(index=date_range, dtype=float, name=f"merged_predictions_series_{variable.lower()}")
    merged_predictions_series.update(train_predict)
    merged_predictions_series.update(test_predict)

    #Generate train residuals
    resid_train = model_results.resid
    resid_train.name = f"{variable.lower()}_resid_train"

    # Compute complete residuals
    resid_complete = transformed_series - merged_predictions_series
    resid_complete.name = f"{variable.lower()}_resid_complete"
    
    # Display results
    display(resid_complete)
    display(resid_train)

    return resid_complete, resid_train


def generate_resid_append(
    variable: str,
    start_date: str,
    end_date: str,
    train,
    test,
    train_predict,
    test_predict,
    model_results
):
    """
    Generate residuals and overwrite initial values of resid_complete with resid_train.
    """
    resid_complete, resid_train = generate_resid(
        variable, start_date, end_date,
        train,
        test,
        train_predict,
        test_predict,
        model_results
    )
    
    # Overwrite initial values of resid_complete with resid_train
    resid_complete.loc[resid_train.index] = resid_train
    
    # Display results
    display(resid_complete)
    
    return resid_complete
    
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

def fit_garch(variable: str, vol: str, p: int, q: int, dist: str, last_obs: str, residuals):
    """
    Fit a GARCH(p, q) model to the residuals.
    """
    if residuals is None:
        raise ValueError(f"Residuals for {variable} are not provided.")
    
    # Check last few values before fitting
    print("Last values in resid_complete before fitting:")
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

def garch_predict(
    variable: str, dist: str, model: str, start_date: str, end_date: str, garch_fitted_model,
    train,
    train_predict,
    test_predict
):
    """
    Generate predictions using a SARIMAX-GARCH or SARIMAX-EGARCH model.
    """
    rs = np.random.RandomState(42)
    
    # Generate GARCH forecasts
    forecast_horizon = len(test_predict)
    garch_forecast = garch_fitted_model.forecast(
        horizon=forecast_horizon, start=start_date, align='origin', reindex=False, method='simulation', simulations=100
    )
    
    forecasted_variance = garch_forecast.variance.iloc[-1, :forecast_horizon].values
    forecasted_stddev = np.sqrt(forecasted_variance)
    
    if dist == "normal":
        simulated_z = rs.normal(loc=0, scale=1, size=forecast_horizon)
    elif dist == "t":
        simulated_z = rs.standard_t(df=garch_fitted_model.params['nu'], size=forecast_horizon)
    elif dist == "ged":
        simulated_z = gennorm.rvs(beta=garch_fitted_model.params['nu'], size=forecast_horizon, random_state=rs)
    else:
        raise ValueError("Invalid distribution. Choose from 'normal', 't', or 'ged'.")
    
    predicted_et = forecasted_stddev * simulated_z
    predicted_et = pd.Series(predicted_et, index=test_predict.index)
    combined_predictions = test_predict + predicted_et
    
    fig, ax = plt.subplots(figsize=(10, 6))
    train.plot(ax=ax, label='Train Data', color='blue')
    test_predict.plot(ax=ax, label='SARIMAX Predictions (mu)', color='orange')
    train_predict.plot(ax=ax, label='Train Set Predictions', color='green')
    combined_predictions.plot(ax=ax, label=f'SARIMAX-{model.upper()} Predictions', color='red')
    ax.set_title(f'Predictions with SARIMAX-{model.upper()} Model ({variable.upper()})')
    ax.legend()
    plt.show()
    
    return combined_predictions

def inverse_transform(boxcoxy_fit_loaded, gwap_predictions, lwap_predictions, test_date):
    """
    Computes inverse-transformed predictions for both 'GWAP' and 'LWAP'.
    
    Parameters:
        boxcoxy_fit_loaded (object): Pre-fitted Box-Cox transformer.
        gwap_combined_predictions (pd.DataFrame): DataFrame containing combined predictions for GWAP.
        lwap_combined_predictions (pd.DataFrame): DataFrame containing combined predictions for LWAP.
    
    Returns:
        tuple: Two pd.Series containing inverse-transformed predictions for 'GWAP' and 'LWAP'.
    """
    
    # Combine predictions
    all_predictions = pd.concat([gwap_predictions, lwap_predictions], axis=1)
    
    # Perform inverse transformation
    all_predictions_inverse = boxcoxy_fit_loaded.inverse_transform(all_predictions)
    
    # Create DataFrame
    all_predictions_inverse_df = pd.DataFrame(all_predictions_inverse)
    all_predictions_inverse_df.rename(columns={'0': 'GWAP', '1': 'LWAP'}, inplace=True)
    
    # Process GWAP
    gwap_predictions_inverse = pd.DataFrame(all_predictions_inverse[:, 0], columns=['GWAP'])
    gwap_predictions_inverse = pd.concat([test_date, gwap_predictions_inverse], axis=1)
    gwap_predictions_inverse.set_index('Date', inplace=True)
    gwap_predictions_inverse.index = pd.to_datetime(gwap_predictions_inverse.index)
    gwap_predictions_inverse.index.freq = 'D'
    gwap_predictions_inverse = gwap_predictions_inverse.squeeze()
    
    # Process LWAP
    lwap_predictions_inverse = pd.DataFrame(all_predictions_inverse[:, 1], columns=['LWAP'])
    lwap_predictions_inverse = pd.concat([test_date, lwap_predictions_inverse], axis=1)
    lwap_predictions_inverse.set_index('Date', inplace=True)
    lwap_predictions_inverse.index = pd.to_datetime(lwap_predictions_inverse.index)
    lwap_predictions_inverse.index.freq = 'D'
    lwap_predictions_inverse = lwap_predictions_inverse.squeeze()
    
    return gwap_predictions_inverse, lwap_predictions_inverse
    
def metrics(data, test_actual: pd.Series, test_predictions: pd.Series, variable: str, model: str):
    """
    Compute MAE, RMSE, and MAPE for given test actual and predicted values.
    """
    mae_test = mean_absolute_error(test_actual, test_predictions)
    mse_test = mean_squared_error(test_actual, test_predictions)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
    
    print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
    print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse_test}")
    print(f"Mean Absolute Percentage Error (MAPE) on Test Set: {mape_test}%")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if variable == 'GWAP':
        data.plot(ax=ax, label='Original Data')
    elif variable == 'LWAP':
        data.plot(ax=ax, label='Original Data')
    test_predictions.plot(ax=ax, label='Predictions on Test Set', linestyle='--')
    ax.set_title(f'SARIMAX-{model.upper()} {variable.upper()} Predictions')
    ax.legend()
    plt.show()
    
    return mae_test, rmse_test, mape_test
