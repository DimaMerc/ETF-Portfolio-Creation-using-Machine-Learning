import numpy as np
from sklearn.preprocessing import StandardScaler

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def check_for_nan_inf(data, name, raise_error=False):
    import numpy as np
    print(f"Checking {name} for NaNs and Infs...")
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"{name} contains {nan_count} NaNs and {inf_count} Infs.")
    if raise_error and (nan_count > 0 or inf_count > 0):
        raise ValueError(f"{name} contains NaNs or Infs.")


def preprocess_data(X_lstm_train, X_gcn_train, y_train, X_lstm_val, X_gcn_val, y_val):
    def check_for_nan_inf(data, name):
        print(f"Checking {name} for NaNs and Infs...")
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        print(f"{name} contains {nan_count} NaNs and {inf_count} Infs.")
        return nan_count > 0 or inf_count > 0

    # Replace NaNs and Infs with reasonable values
    for data, name in [(X_lstm_train, 'X_lstm_train'), (X_gcn_train, 'X_gcn_train'), (y_train, 'y_train'),
                       (X_lstm_val, 'X_lstm_val'), (X_gcn_val, 'X_gcn_val'), (y_val, 'y_val')]:
        if check_for_nan_inf(data, name):
            data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale LSTM data
    num_samples_train, time_steps, features = X_lstm_train.shape
    num_samples_val = X_lstm_val.shape[0]
    scaler_lstm = StandardScaler()
    X_lstm_train = scaler_lstm.fit_transform(X_lstm_train.reshape(-1, features)).reshape(num_samples_train, time_steps, features)
    X_lstm_val = scaler_lstm.transform(X_lstm_val.reshape(-1, features)).reshape(num_samples_val, time_steps, features)

    # Scale GCN data
    scaler_gcn = StandardScaler()
    X_gcn_train = scaler_gcn.fit_transform(X_gcn_train)
    X_gcn_val = scaler_gcn.transform(X_gcn_val)

    # Scale target data
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    return X_lstm_train, X_gcn_train, y_train, X_lstm_val, X_gcn_val, y_val
