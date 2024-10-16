# ml_model_module.py
# ml_model_module.py
import keras
from keras import layers
import tensorflow as tf

class CustomModel(keras.Model):
    def __init__(self, lstm_input_shape, gcn_input_shape, **kwargs):
        super().__init__(**kwargs)
        self.lstm_input_shape = lstm_input_shape
        self.gcn_input_shape = gcn_input_shape
        
        self.lstm_input = layers.Input(shape=lstm_input_shape, name='lstm_input')
        self.gcn_input = layers.Input(shape=gcn_input_shape, name='gcn_input')

        # LSTM branch
        x_lstm = layers.LSTM(128, return_sequences=True)(self.lstm_input)
        x_lstm = layers.LSTM(64, return_sequences=True)(x_lstm)
        x_lstm = layers.LSTM(32)(x_lstm)

        # GCN branch
        x_gcn = layers.Dense(64, activation='relu')(self.gcn_input)
        x_gcn = layers.Dense(32, activation='relu')(x_gcn)

        # Concatenate
        x = layers.Concatenate()([x_lstm, x_gcn])

       # x = layers.Dense(128, activation='relu')(x)
       # x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        output = layers.Dense(1)(x)

        self.model = keras.Model(inputs={'lstm_input': self.lstm_input, 'gcn_input': self.gcn_input}, outputs=output)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.mae_metric = keras.metrics.MeanAbsoluteError(name='mae')

    def call(self, inputs):
        return self.model(inputs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        
        return {'loss': self.loss_tracker.result(), 'mae': self.mae_metric.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        
        return {'loss': self.loss_tracker.result(), 'mae': self.mae_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def get_config(self):
        config = super().get_config()
        config.update({
            "lstm_input_shape": self.lstm_input_shape,
            "gcn_input_shape": self.gcn_input_shape,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)