import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

def logistic_regress_out(df, y, idx=None):
    if idx is None:
        idx = y.index
        
    X = df.loc[idx]
    y_ = y.loc[idx]
    
    model = LogisticRegression(penalty=None, solver='newton-cholesky')
    model.fit(X, y_)
    
    lin_pred = model.decision_function(X)
    probas = model.predict_proba(X)[:, 1]

    lin_pred = pd.Series(lin_pred, index=idx)
    probas = pd.Series(probas, index=idx)

    return lin_pred, probas


def lin_regress_out(df, y, idx=None):
    if idx is None:
        idx = y.index
        
    X = df.loc[idx]
    y_ = y.loc[idx]
    
    model = LinearRegression()
    model.fit(X, y_)
    
    preds = model.predict(df.loc[idx])
    res = y - preds

    return res, preds


def impute_covars(df, idx=None):
    df_covars = df.copy()
    covar_cols = df_covars.columns
    if idx is None:
        idx = df_covars.index
    
    n_missing = df_covars.isnull().sum()
    print("N Missing Data")
    print(n_missing)
    
    cat_cols = []
    num_cols = []
    
    for col in covar_cols:
        unique_vals = df_covars.loc[idx, col].nunique(dropna=True)
        if unique_vals < 5:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    
    mode_vals = df_covars.loc[idx, cat_cols].mode().iloc[0]
    avg_vals = df_covars.loc[idx, num_cols].mean()
    
    df_covars.loc[:, cat_cols] = df_covars[cat_cols].fillna(mode_vals)
    df_covars.loc[:, num_cols] = df_covars[num_cols].fillna(avg_vals)

    return df_covars

def standardize_covars(df, idx=None):
    df_covars = df.copy()
    if idx is None:
        idx = df_covars.index
    
    df_covars_mean = df_covars.loc[idx].mean()
    df_covars_std = df_covars.loc[idx].std()

    df_covars = (df_covars - df_covars_mean) / df_covars_std

    return df_covars, df_covars_mean, df_covars_std
    