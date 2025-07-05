from tensorflow.keras import layers,Input,Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from model_train import get_timestep_embedding

### Old Model 
def get_transformer_denoise_model(embedding_dim,input_dim):
    y_t_input = Input(shape=(1,))
    t_input = Input(shape=(embedding_dim,))
    nyse_input = Input(shape=(input_dim,))

    x = layers.Reshape((input_dim, 1))(nyse_input)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Concatenate()([x, y_t_input, t_input])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1)(x)

    return Model([y_t_input, t_input, nyse_input], x)

### New Model 
# def get_transformer_denoise_model(embedding_dim,input_dim):
#     y_t_input = Input(shape=(1,))
#     t_input = Input(shape=(embedding_dim,))
#     nyse_input = Input(shape=(input_dim,))

#     # Step 1: Reshape NYSE input for sequence processing
#     nyse_reshaped = layers.Reshape((input_dim, 1))(nyse_input)  # (None, 30, 1)

#     # Step 2: Frozen positional embeddings
#     position = tf.range(start=0, limit=input_dim, delta=1)
#     pos_emb = get_timestep_embedding(position, dim=8)  # (30, 8)
#     pos_emb = tf.expand_dims(pos_emb, axis=0)  # (1, 30, 8)
#     pos_emb_layer = tf.constant(pos_emb.numpy())

#     def repeat_pos_emb(x):
#         batch_size = tf.shape(x)[0]
#         return tf.tile(pos_emb_layer, [batch_size, 1, 1])

#     pos_emb_tensor = layers.Lambda(repeat_pos_emb)(nyse_reshaped)

#     # Step 3: Concatenate NYSE input with positional embeddings
#     x = layers.Concatenate(axis=-1)([nyse_reshaped, pos_emb_tensor])  # (None, 30, 9)

#     # Step 4: Conv1D + LayerNorm
#     x = layers.Conv1D(
#         filters=128,
#         kernel_size=3,
#         padding='same',
#         activation='relu',
#         kernel_regularizer=l2(1e-4)
#     )(x)
#     x = layers.LayerNormalization()(x)

#     # Step 5: Transformer Self-Attention with Residual
#     residual = x  # Save for residual connection
#     attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
#     x = layers.Add()([residual, attention_out])  # Residual connection
#     x = layers.LayerNormalization()(x)

#     # Optional Dropout (after attention+residual)
#     x = layers.Dropout(0.1)(x)

#     # Step 6: Global pooling
#     x = layers.GlobalAveragePooling1D()(x)

#     # Step 7: Combine with y_t and t embeddings
#     x = layers.Concatenate()([x, y_t_input, t_input])
#     x = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
#     x = layers.Dense(1, kernel_regularizer=l2(1e-4))(x)

#     return Model([y_t_input, t_input, nyse_input], x)


# def get_Residual_LSTM_denoise_model(embedding_dim,input_dim):
#     y_t_input = Input(shape=(1,))
#     t_input = Input(shape=(embedding_dim,))
#     nyse_input = Input(shape=(input_dim,))

#     t_emb = layers.Dense(32, activation='relu')(t_input)
#     x = layers.Concatenate()([t_emb, nyse_input])
#     x = layers.Reshape((1, 62))(x)

#     lstm_out = layers.LSTM(64, return_sequences=True)(x)
#     skip = lstm_out  # Save for residual

#     lstm_out = layers.LSTM(64, return_sequences=False)(lstm_out)
#     x = layers.Concatenate()([lstm_out, y_t_input])
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(1)(x)

#     return Model(inputs=[y_t_input, t_input, nyse_input], outputs=x)

#new lstm model
def get_Residual_LSTM_denoise_model(embedding_dim, input_dim):
    y_t_input = Input(shape=(1,))
    t_input = Input(shape=(embedding_dim,))
    nyse_input = Input(shape=(input_dim,))

    # Step 1: Reshape input for LSTM
    x = layers.Reshape((input_dim, 1))(nyse_input)

    # Step 2: Add Positional Embeddings
    position = tf.range(start=0, limit=input_dim, delta=1)
    pos_emb = get_timestep_embedding(position, dim=8)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    pos_emb_layer = tf.constant(pos_emb.numpy())

    def repeat_pos_emb(x_):
        batch_size = tf.shape(x_)[0]
        return tf.tile(pos_emb_layer, [batch_size, 1, 1])

    pos_emb_tensor = layers.Lambda(repeat_pos_emb)(x)
    x = layers.Concatenate(axis=-1)([x, pos_emb_tensor])  # (batch, 30, 9

    # Step 3: Bidirectional LSTM + Residual
    x_skip = x
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, layers.GlobalAveragePooling1D()(x_skip)])  # Residual

    # Step 4: Combine with y_t and t
    x = layers.Concatenate()([x, y_t_input, t_input])
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dense(1, kernel_regularizer=l2(1e-4))(x)

    return Model(inputs=[y_t_input, t_input, nyse_input], outputs=x)
