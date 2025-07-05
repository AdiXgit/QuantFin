import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

# Timestep 
def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = tf.exp(
        -tf.math.log(10000.0) * tf.range(0, half, dtype=tf.float32) / (half - 1)
    )
    angles = tf.cast(timesteps, tf.float32)[:, None] * freqs[None, :]
    emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=1)
    return emb

# Forward Pass
def q_sample(y_0,t,noise,alpha_bars):
    a_bar = tf.gather(alpha_bars,t)
    a_bar = tf.reshape(a_bar,(-1,1))
    return tf.sqrt(a_bar)*y_0+tf.sqrt(1.-a_bar)*noise


#  Training Step
@tf.function
def train_step(model,optimizer,x_batch,y_batch,embedding_dim,alpha_bars,T):
    batch_size = tf.shape(x_batch)[0]
    t = tf.random.uniform((batch_size,),minval=0,maxval=T,dtype=tf.int32)
    t_emb = get_timestep_embedding(t,embedding_dim)

    noise = tf.random.normal(shape=tf.shape(y_batch))
    y_t = q_sample(y_batch,t,noise,alpha_bars)

    with tf.GradientTape() as tape:
        pred_noise = model([y_t,t_emb,x_batch],training=True)
        loss = tf.reduce_mean(tf.square(pred_noise-noise))
    
    grads = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    
    return loss

#  Train Loop
def training_model(train_dataset,model,embedding_dim,alpha_bars,T): 
    num_epochs = 1000
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=100,
        decay_rate=0.98,
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)

    print(f"Training Model")
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dataset:
            loss = train_step(model, optimizer, x_batch, y_batch,embedding_dim,alpha_bars,T)
        print(f"Epoch {epoch+1}: Loss = {loss.numpy():.4f}")

        