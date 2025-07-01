import tensorflow as tf 
import numpy as np

from preprocess_data import get_data
from denoise_models import get_transformer_denoise_model,get_Residual_LSTM_denoise_model
from model_train import training_model
from model_prediction import predict,predict_next_day_from_validation

# Hyperparameters
T = 1000
beta_start = 1e-4
beta_end = 0.02
embedding_dim = 32
input_dim = 30

# Noise Schedule
betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars_np = np.cumprod(alphas).astype(np.float32)
alpha_bars = tf.convert_to_tensor(alpha_bars_np, dtype=tf.float32)

# Denoise Models
model_transformer = get_transformer_denoise_model(embedding_dim,input_dim)
# model_res_lstm = get_Residual_LSTM_denoise_model(embedding_dim,input_dim)

# Preprocessed Data
train_dataset, x_val_batch, nse_scaler, nyse_scaler,x_val, y_val = get_data()

# Training 
training_model(train_dataset,model_transformer,embedding_dim,alpha_bars,T)
# training_model(train_dataset,model_res_lstm,Adam(1e-3),embedding_dim,alpha_bars,T)

mse, mae, mse_drift, mae_drift, direction_accuracy_drift = predict(model_transformer,x_val_batch,nse_scaler,y_val,embedding_dim,alpha_bars,T)
print(f"Transformer \nMSE : {mse:0.6f},MAE : {mae:0.6f}")
print(f"Drfit-Adjusted MSE : {mse_drift:0.6f},MAE : {mae_drift:0.6f}")
print(f"Drift-Adjusted Directional Accuracy: {direction_accuracy_drift:.2f}%")

# mse, mae, mse_drift, mae_drift, direction_accuracy_drift = predict(model_res_lstm,x_val_batch,nse_scaler,y_val,embedding_dim,alpha_bars,T)
# print(f"LSTM \nMSE : {mse:0.6f},MAE : {mae:0.6f}")
# print(f"Drfit-Adjusted MSE : {mse_drift:0.6f},MAE : {mae_drift:0.6f}")
# print(f"Drift-Adjusted Directional Accuracy: {direction_accuracy_drift:.2f}%")


pred_unscaled, pred_drifted, direction = predict_next_day_from_validation(
    model_transformer, x_val, y_val, nse_scaler, embedding_dim, alpha_bars, T
)


