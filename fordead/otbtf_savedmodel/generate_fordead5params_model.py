import tensorflow as tf
import math

# Inputs
series = tf.keras.Input(shape=[None, None, None], name="ts")  # [1, h, w, N]
dates = tf.keras.Input(shape=[], name="dates")  # [N]

# Harmonic terms
pi = tf.constant(math.pi)
fac = tf.multiply(dates, 2.0 * pi / 365.25)
harmonic_terms = [tf.ones_like(fac), tf.math.sin(fac), tf.math.cos(fac), tf.math.sin(2 * fac), tf.math.cos(2 * fac)]
A = tf.stack(harmonic_terms, axis=-1)  # [N, R] with R=5

# No-data masks
masks = tf.cast(tf.math.greater(series, tf.zeros_like(series)), tf.float32)

# Reshape pixel blocks as matrices
N = tf.shape(dates)[0]
m = tf.shape(masks)[1] * tf.shape(masks)[2]
At = tf.transpose(A)  # [R, N]
B = tf.reshape(series, shape=[m, N])  # [1, h, w, N] --> [m, N]
M = tf.reshape(masks, shape=[m, N])  # [1, h, w, N] --> [m, N]

# Avoid pixels with too few observations
rank_ok = tf.math.greater(tf.reduce_sum(M, axis=-1), 5.0 * tf.ones([m]))
idx_keep = tf.where(rank_ok)  # [m]
idx_keep = tf.reshape(idx_keep, shape=[-1])
idx_keep = tf.cast(idx_keep, tf.int64)
B = tf.gather(B, indices=idx_keep, axis=0)
M = tf.gather(M, indices=idx_keep, axis=0)

# Model fit using least square regression with missing data
# see http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
MB = tf.transpose(tf.multiply(M, B))
prod = tf.tensordot(At, MB, axes=((1,), (0,)))
rhs = tf.expand_dims(tf.transpose(prod), axis=-1)
MA = tf.multiply(tf.expand_dims(M, axis=-1), tf.expand_dims(A, axis=0))
T = tf.linalg.matmul(tf.expand_dims(At, axis=0), MA)
coefficients = tf.squeeze(tf.linalg.lstsq(T, rhs, fast=False), axis=-1)

# Replace outputs at right indices and reform pixel block
idx_keep = tf.expand_dims(idx_keep, axis=-1)
out = tf.scatter_nd(idx_keep, shape=[m, 5], updates=coefficients)
coefs_out = tf.reshape(out, shape=[1, tf.shape(masks)[1], tf.shape(masks)[2], 5])

# Create model
model = tf.keras.Model(inputs={"ts": series, "dates": dates}, outputs={"coefs": coefs_out})
model.save("fordeadparams_model")
