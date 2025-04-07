from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import root_scalar
import numpy as np

def efficient_logistic_regression(x, y, fetch_params=None, n_jobs=-1):
    if fetch_params is None:
        fetch_params = ['preds', 'weights', 'bias', 'log_likelihood', 'auc', 'converged']
    
    if x.ndim == 2 and y.ndim == 1:
        x = x[None, ...]
        y = y[None, ...]

    n_runs = x.shape[0]
    
    def fit_logistic_regression(x_run, y_run, do_auc=False):
        model = LogisticRegression(
            fit_intercept=True, 
            solver='saga',
            max_iter=500,
            penalty=None
        )
        model.fit(x_run, y_run)
        weights = model.coef_
        bias = model.intercept_
        preds = model.predict_proba(x_run)[:, 1]
        
        log_likelihood = -log_loss(y_run, preds, labels=[0, 1]) * len(y_run)
        
        auc = None
        if do_auc:
            auc = roc_auc_score(y_run, preds)
        
        converged = model.n_iter_[0] < model.max_iter  # Check if it converged
        
        return {
            'weights': weights.flatten(),
            'bias': bias.item(),
            'preds': preds,
            'log_likelihood': log_likelihood,
            'auc': auc,
            'converged': converged
        }

    if n_jobs == -1:  # Use a simple for loop when n_jobs is -1
        results_list = [
            fit_logistic_regression(x[i], y[i], 'auc' in fetch_params) for i in range(n_runs)
        ]
    else:  # Use parallel processing when n_jobs is specified
        from joblib import Parallel, delayed
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(fit_logistic_regression)(x[i], y[i], 'auc' in fetch_params) for i in range(n_runs)
        )
    
    results = {key: np.array([res[key] for res in results_list]) for key in fetch_params}
    return results

def efficient_linear_regression(x, y, fetch_params=None):
    if fetch_params is None:
        fetch_params = ['preds', 'residuals', 'weights', 'bias', 'r2']
    
    if x.ndim == 2 and y.ndim == 1:
        x = x[None, ...]
        y = y[None, ...]
    
    # Add intercept column to x
    x_augmented = np.concatenate([x, np.ones((x.shape[0], x.shape[1], 1))], axis=2)  # [n_runs, n_samples, n_features+1]
    
    # Compute regression coefficients
    coeffs = np.linalg.solve(
        np.matmul(x_augmented.transpose(0, 2, 1), x_augmented),  # [n_runs, n_features+1, n_features+1]
        np.matmul(x_augmented.transpose(0, 2, 1), y[..., np.newaxis])  # [n_runs, n_features+1, 1]
    )  # [n_runs, n_features+1, 1]
    
    # Compute predictions
    y_preds = np.matmul(x_augmented, coeffs).squeeze(-1)  # [n_runs, n_samples]
    
    # Compute residuals
    residuals = None
    if 'residuals' in fetch_params:
        residuals = y - y_preds  # [n_runs, n_samples]

    # Compute R-squared
    r2 = None
    if 'r2' in fetch_params:
        total_variance = np.sum((y - y.mean(axis=1, keepdims=True))**2, axis=1)
        residual_variance = np.sum((residuals if residuals is not None else y - y_preds)**2, axis=1)
        r2 = 1 - (residual_variance / total_variance)  # [n_runs]
    
    results = {
        'preds': y_preds,
        'residuals': residuals,
        'weights': coeffs.squeeze(-1)[:, :-1],
        'bias': coeffs.squeeze(-1)[:, -1],
        'r2': r2
    }
    
    return {key: results[key] for key in fetch_params}

def efficient_r2(x, y):
    if x.ndim == 1 and y.ndim == 1:
        x = x[None, ...]
        y = y[None, ...]

    mean_x = np.mean(x, axis=1, keepdims=True)
    mean_y = np.mean(y, axis=1, keepdims=True)

    x_centered = x - mean_x
    y_centered = y - mean_y
    
    cov_xy = np.sum(x_centered * y_centered, axis=1) / (x.shape[1] - 1)
    var_x = np.sum(x_centered ** 2, axis=1) / (x.shape[1] - 1)
    var_y = np.sum(y_centered ** 2, axis=1) / (x.shape[1] - 1)
    
    weight = cov_xy / var_x
    r2 = (cov_xy / np.sqrt(var_x * var_y)) ** 2

    results = {'r2': r2,
               'weight': weight}
    
    return results

def effecient_r2_to_risk(r2_obs, prev, top_quant=0.1, quant_val=None):
    # Precompute constants
    d = np.sqrt((prev + (1 - prev))**2 / (prev * (1 - prev))) * np.sqrt(r2_obs) / np.sqrt(1 - r2_obs)
    mu_case = d
    mu_control = 0
    varPRS = prev * (1 + d**2 - (d * prev)**2) + (1 - prev) * (1 - (d * prev)**2)
    E_PRS = d * prev

    quants = np.asarray([1-top_quant, top_quant])
    
    # Solve for quantile thresholds using vectorized operations
    def quant_f_solve(x, p, d_val):
        return p * norm.cdf(x - d_val) + (1 - p) * norm.cdf(x) - (1 - top_quant)

    ul_qv_PRS = []
    for i, (p, d_val) in enumerate(zip(prev, d)):
        if quant_val is None:
            solution = root_scalar(quant_f_solve, args=(p, d_val), bracket=(-2.5, 2.5),  method='brentq', xtol=1e-8) # 6e-12
        elif quant_val:
            try:
                solution = root_scalar(quant_f_solve, args=(p, d_val), bracket=(quant_val-0.05, quant_val+0.05),  method='brentq', xtol=1e-8) # 6e-12
            except ValueError:
                solution = root_scalar(quant_f_solve, args=(p, d_val), bracket=(-2.5, 2.5),  method='brentq', xtol=1e-8) # 6e-12s

        quant_val = solution.root if solution.converged else np.nan
        quant_val_adj = (quant_val-E_PRS[i])/np.sqrt(varPRS[i])
        ul_qv_PRS += [np.array([[-np.inf, quant_val, -np.inf, quant_val_adj], 
                                [quant_val, np.inf, quant_val_adj, np.inf]])]
    ul_qv_PRS = np.asarray(ul_qv_PRS)
        
    # Compute probabilities for cases and controls
    prob_case = norm.cdf(ul_qv_PRS[:, :, 1], loc=mu_case[:,None]) - norm.cdf(ul_qv_PRS[:, :, 0], loc=mu_case[:,None])
    prob_control = norm.cdf(ul_qv_PRS[:, :, 1], loc=mu_control) - norm.cdf(ul_qv_PRS[:, :, 0], loc=mu_control)

    # Calculate probabilities normalized to the top quantile
    p_case = (prob_case * prev[:,None]) / quants[None,:]
    p_control = (prob_control * (1 - prev[:,None])) / quants[None,:]

    # Compute odds ratios and absolute risks
    OR = (p_case / p_control)  # Last segment's OR
    OR = OR/OR[:,:1]
    OR = OR[:,-1]
    AR = p_case[:, -1]  # Last segment's absolute risk
    AR_control = p_case[:, 0]

    # Combine results into a dictionary
    results = {'odds_ratio': OR, 
               'abs_risk': AR, 
               'abs_risk_control': AR_control, 
               'quant_val': quant_val}
    return results

def efficient_odds_ratio(r2_obs, prev):
    p = prev
    d_vals = np.sqrt((p + (1 - p))**2 / (p * (1 - p))) * np.sqrt(r2_obs) / np.sqrt(1 - r2_obs)
    odds_ratio = np.exp(d_vals * np.sqrt(1 + d_vals**2 * p * (1 - p)))
    return odds_ratio

def efficient_auc(r2_obs, prev):
    p = prev
    d_vals = np.sqrt((p + (1 - p))**2 / (p * (1 - p))) * np.sqrt(r2_obs) / np.sqrt(1 - r2_obs)
    auc = auc = norm.cdf(abs(d_vals) / math.sqrt(2), 0, 1)
    return auc

def effecient_r2_liab(r2_obs, prev):
    P = K = prev

    thd = norm.ppf(1 - K)
    zv = np.exp(-0.5 * thd**2) / np.sqrt(2 * np.pi)
    
    mv = zv / K
    mv2 = -mv * K / (1 - K)
    
    theta = mv * (P - K) / (1 - K) * (mv * (P - K) / (1 - K) - thd)
    cv = K * (1 - K) / zv**2 * K * (1 - K) / (P * (1 - P))
    
    r2_liab = r2_obs * cv / (1 + r2_obs * theta * cv)

    return r2_liab