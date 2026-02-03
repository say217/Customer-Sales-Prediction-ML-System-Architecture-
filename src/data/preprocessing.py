def remove_outliers_iqr(X, y, multiplier=1.5):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    mask = (y >= lower) & (y <= upper)
    return X.loc[mask], y.loc[mask]
