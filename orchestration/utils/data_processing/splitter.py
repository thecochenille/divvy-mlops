import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, params):
    features = df.drop(['net_usage'], axis =1)
    target = df['net_usage']

    X_train, X_test, y_train, y_test = train_test_split(features, target, **params)

    return X_train, X_test, y_train, y_test