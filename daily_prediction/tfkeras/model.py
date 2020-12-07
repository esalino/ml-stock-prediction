import tensorflow as tf


def create_model(hidden_dim, output_dim, sequence_length, input_dim):
    model = tf.keras.models.Sequential()

    #model.add(tf.keras.layers.GRU(hidden_dim, return_sequences=True, input_shape=(sequence_length, input_dim)))
    model.add(tf.keras.layers.GRU(hidden_dim))
    model.add(tf.keras.layers.Dense(output_dim))
    model.add(tf.keras.layers.Activation('linear'))

    model.compile(loss='mse', optimizer='Adam')

    return model
