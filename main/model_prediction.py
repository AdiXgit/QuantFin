from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from model_train import get_timestep_embedding

#  DDIM Sampling
def ddim_sample(model,x_cond,embedding_dim,alpha_bars,T,eta=0.0):
    batch_size = tf.shape(x_cond)[0]
    y_t = tf.random.normal((batch_size, 1))

    for i in reversed(range(1, T)):
        batch_size = tf.shape(x_cond)[0]
        t = tf.fill([batch_size], i)  # shape (234,)
        t_emb = get_timestep_embedding(t, embedding_dim)

        pred_noise = model([y_t, t_emb, x_cond], training=False)

        a_bar = tf.gather(alpha_bars, t)
        a_bar_prev = tf.gather(alpha_bars, t-1)

        a_bar = tf.reshape(a_bar, (-1, 1))
        a_bar_prev = tf.reshape(a_bar_prev, (-1, 1))

        y_0 = (y_t - tf.sqrt(1. - a_bar) * pred_noise) / tf.sqrt(a_bar)
        y_t = tf.sqrt(a_bar_prev) * y_0 + tf.sqrt(1. - a_bar_prev) * pred_noise

    return y_0

def predict(model,x_val_batch,nse_scaler,y_val,embedding_dim,alpha_bars,T):
    preds = ddim_sample(model,x_val_batch,embedding_dim,alpha_bars,T)
    preds = preds.numpy()
    preds = nse_scaler.inverse_transform(preds)

    true_y = nse_scaler.inverse_transform(y_val)

    mse = mean_squared_error(true_y, preds)
    mae = mean_absolute_error(true_y, preds)

    # Drift Adjustment
    nse_trend = np.mean(true_y)  
    preds_with_drift = preds + nse_trend

    # Recalculate Metrics 
    mse_drift = mean_squared_error(true_y, preds_with_drift)
    mae_drift = mean_absolute_error(true_y, preds_with_drift)

    # Directional Accuracy with Drift 
    direction_accuracy_drift = np.mean(np.sign(true_y) == np.sign(preds_with_drift)) * 100

    return mse, mae, mse_drift, mae_drift, direction_accuracy_drift

    # Plot 
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 4))
    # plt.plot(true_y, label='Actual NSE Returns', linewidth=1)
    # plt.plot(preds_with_drift, label='Predicted NSE Returns (with Drift)', linewidth=1)
    # plt.title("DDIM Predicted vs Actual NSE Returns (Drift Adjusted)")
    # plt.legend()
    # plt.show()

def predict_next_day_from_validation(model, x_val, y_val, nse_scaler, embedding_dim, alpha_bars, T):
    # Get the most recent conditioning window
    x_input = x_val[-1].reshape(1, -1).astype(np.float32)

    # DDIM Sampling
    x_tensor = tf.convert_to_tensor(x_input)
    pred = ddim_sample(model, x_tensor, embedding_dim, alpha_bars, T)
    pred_unscaled = nse_scaler.inverse_transform(pred.numpy())

    # Drift Adjustment using overall y_val (validation NSE)
    nse_trend = np.mean(nse_scaler.inverse_transform(y_val))
    pred_drifted = pred_unscaled + nse_trend

    # Direction: +1 = Up, -1 = Down, 0 = Flat
    direction = int(np.sign(pred_drifted[0, 0]))

    print("=== Prediction for Next Day ===")
    print(f"NSE Return (Raw)           : {pred_unscaled[0, 0]:.6f}")
    print(f"NSE Return (Drift Adjusted): {pred_drifted[0, 0]:.6f}")
    print(f"Predicted Direction        : {'Up' if direction == 1 else 'Down' if direction == -1 else 'Flat'}")

    return pred_unscaled[0, 0], pred_drifted[0, 0], direction



