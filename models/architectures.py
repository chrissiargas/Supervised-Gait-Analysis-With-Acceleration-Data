import tensorflow as tf

from keras.layers import (Input, LSTM, Conv2D, Conv1D, Dense,
                          Dropout, GlobalMaxPool1D, Activation, ReLU,
                          MaxPooling1D, ZeroPadding1D, Flatten, LayerNormalization, Layer,
                          BatchNormalization, Multiply, Bidirectional, Permute, Lambda, RepeatVector,
                          GRU, ZeroPadding2D, MaxPooling2D)
from keras.models import Model
from keras import initializers, Sequential
import keras.backend as K
from keras.src.layers import concatenate
from tcn import TCN
import keras
from typing import Optional
from preprocessing.utils import impute

class conv1d_block(Layer):
    def __init__(self,
                 use_dropout: bool = False,
                 kernel_size: int = 3,
                 filters: int = 64,
                 use_pooling: bool = True,
                 use_norm: bool = False):
        super().__init__()

        layers = []

        if use_norm:
            layers.append(BatchNormalization())
        layers.append(ZeroPadding1D(padding=1))
        layers.append(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='valid',
                             kernel_initializer=initializers.he_normal()
        ))
        layers.append(ReLU())
        if use_dropout:
            layers.append(Dropout(0.4))
        if use_pooling:
            layers.append(MaxPooling1D(pool_size=2, strides=2))

        self.conv_net = Sequential(layers)

    def call(self, inputs):
        y = self.conv_net(inputs)
        return y


class CNNGRU_encoder(Layer):
    def __init__(self,  n_units: int):
        super().__init__()

        self.conv1d_blocks = [conv1d_block(
            filters=n_units,
            kernel_size=3,
            use_pooling=False,
            use_norm=True,
            use_dropout=False
        ) for _ in range(3)]

        self.grus = [
            Bidirectional(GRU(n_units//2,
                              activation='tanh',
                              return_sequences=True))
            for _ in range(2)
        ]

    def call(self, inputs):
        x = inputs

        if self.conv1d_blocks:
            for conv1d in self.conv1d_blocks:
                x = conv1d(x)

        if self.grus:
            for gru in self.grus:
                x = gru(x)

        return x

class channel_attention(keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(channel_attention, self).__init__()
        self.conv_1 = Conv2D(n_filters, kernel_size=kernel_size,
                             padding='same', activation='relu', dilation_rate=dilation_rate)
        self.conv_f = Conv2D(1, kernel_size=1, padding='same')
        self.ln = LayerNormalization()

    def call(self, x):
        x = self.ln(x)
        x1 = tf.expand_dims(x, axis=3)
        x1 = self.conv_1(x1)
        x1 = self.conv_f(x1)
        x1 = tf.keras.activations.softmax(x1, axis=2)
        x1 = tf.keras.layers.Reshape(x.shape[-2:])(x1)

        return tf.math.multiply(x, x1), x1


class positional_encoding(Layer):

    def __init__(self, n_timesteps, n_features):
        super(positional_encoding, self).__init__()
        self.pos_encoding = self.positional_encoding(n_timesteps, n_features)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, n_timesteps, n_features):
        angle_rads = self.get_angles(
            position=tf.range(n_timesteps, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(n_features, dtype=tf.float32)[tf.newaxis, :],
            d_model=n_features)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHead_attention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHead_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class transformer_layer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(transformer_layer, self).__init__()

        self.mha = MultiHead_attention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


class global_temporal_attention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False):
        super(global_temporal_attention, self).__init__()

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.u_constraint = tf.keras.constraints.get(u_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x):
        uit = tf.tensordot(x, self.W, axes=1)

        if self.bias:
            uit += self.b

        uit = tf.keras.activations.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=1)

        a = tf.math.exp(ait)

        a /= tf.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
                     tf.keras.backend.floatx())

        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        result = tf.keras.backend.sum(weighted_input, axis=1)

        if self.return_attention:
            return result, a
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value],
                                  [input_shape[0].value, input_shape[1].value])
        else:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value])

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def get_attention_encoder(input_shape, name: str = 'attention_encoder'):
    inputs = Input(shape=input_shape)
    x = inputs

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = positional_encoding(n_timesteps=input_shape[0], n_features=128)(x)
    x = Dropout(rate=0.1)(x)

    for _ in range(6):
        x = transformer_layer(d_model=128, num_heads=8, dff=256, rate=0.1)(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)

    return Model(inputs, x, name = name)

def get_tcn_encoder(input_shape, name: str = 'tcn_encoder'):
    inputs = Input(shape=(None, input_shape[-1]))
    print(inputs)
    tcn_net = TCN(
        nb_filters=16,
        kernel_size=5,
        dilations=[1, 2, 4],
        padding="same",
        use_batch_norm=True,
        use_skip_connections=True,
        return_sequences=True,
        name="tcn_layer"
    )
    x = tcn_net(inputs)

    return Model(inputs, x, name = name)

