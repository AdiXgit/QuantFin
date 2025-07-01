import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def get_data():
    nifty = pd.read_csv("E:/Coding/Projects/QuantFin/main/data/NIFTY_50_20_Year_His_Data.csv")
    nyse = pd.read_csv("E:/Coding/Projects/QuantFin/main/data/NYSE_20_Year_His_Data.csv")
    
    df_nifty = pd.DataFrame(nifty)
    df_nyse = pd.DataFrame(nyse)

    df_nifty['Date'] = pd.to_datetime(df_nifty['Date'],format='%Y-%m-%d')
    df_nyse['Date'] = pd.to_datetime(df_nyse['Date'],format='%Y-%m-%d')

    df_nifty.set_index('Date', inplace=True)
    df_nyse.set_index('Date', inplace=True)

    common_dates = df_nifty.index.intersection(df_nyse.index)
    df_nifty_common = df_nifty.loc[common_dates]
    df_nyse_common = df_nyse.loc[common_dates]

    df_nifty_common['return'] = df_nifty_common['Close'].pct_change()
    df_nyse_common['return'] = df_nyse_common['Close'].pct_change()

    # Drop NaNs from pct_change
    returns_nifty = df_nifty_common['return'].dropna()
    returns_nyse = df_nyse_common['return'].dropna()

    # Align shapes
    min_len = min(len(returns_nifty), len(returns_nyse))
    returns_nifty = returns_nifty[-min_len:]
    returns_nyse = returns_nyse[-min_len:]

    # Build datasets
    lookback = 30
    x_data = []
    y_data = []
    for i in range(lookback, len(returns_nyse)):
        x_data.append(returns_nyse.iloc[i - lookback:i].values)
        y_data.append([returns_nifty.iloc[i]])

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    # Train-test split
    split = int(0.8 * len(x_data))
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    nyse_scaler = StandardScaler()
    nse_scaler = StandardScaler()

    x_train = nyse_scaler.fit_transform(x_train)
    x_val = nyse_scaler.transform(x_val)

    y_train = nse_scaler.fit_transform(y_train)
    y_val = nse_scaler.transform(y_val)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)
    x_val_batch = tf.convert_to_tensor(x_val, dtype=tf.float32)

    return train_dataset, x_val_batch, nse_scaler, nyse_scaler,x_val, y_val