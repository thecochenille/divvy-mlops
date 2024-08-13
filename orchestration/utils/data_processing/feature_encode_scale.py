import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def encode_categorical(df):
    cat_features = ['station_name', 'day_of_week']
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_cols = pd.DataFrame(ohe.fit_transform(df[cat_features]).toarray(), columns = ohe.get_feature_names_out(cat_features))
    ohe_cols.index = df.index

    encoded_df = pd.concat([df, ohe_cols], axis=1)
    encoded_df = encoded_df.drop(cat_features, axis =1)

    return ohe, encoded_df


def scale_numerical(X_train, X_test, y_train, y_test):
    num_features= ['hour']
    Standard_Scaler = StandardScaler()
    num_scaled_train = pd.DataFrame(Standard_Scaler.fit_transform(X_train[num_features]), columns=['hour_scaled'])
    num_scaled_test = pd.DataFrame(Standard_Scaler.transform(X_test[num_features]), columns=['hour_scaled'])

    num_scaled_train.index = X_train.index
    num_scaled_test.index = X_test.index

    scaled_X_train = pd.concat([X_train, num_scaled_train], axis=1)
    scaled_X_test = pd.concat([X_test, num_scaled_test], axis=1)

    scaled_X_train = scaled_X_train.drop(num_features, axis=1)
    scaled_X_test = scaled_X_test.drop(num_features, axis=1)

    return Standard_Scaler, scaled_X_train, scaled_X_test, y_train, y_test
