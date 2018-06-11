from keras import Input
from keras.engine import Model
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Reshape, GRU, add, concatenate, Lambda
from keras.utils import plot_model

from models.TrainingConfiguration import TrainingConfiguration
from models.loss import ctc_lambda_func


class SimpleConfiguration(TrainingConfiguration):

    def __init__(self, width: int, height: int, alphabet_length: int,
                 maximum_number_of_characters_in_longest_text_line: int):
        super().__init__(data_shape=(width, height, 1))
        self.maximum_number_of_characters_in_longest_text_line = maximum_number_of_characters_in_longest_text_line
        self.alphabet_length = alphabet_length

    def model(self) -> Model:
        act = 'relu'
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512

        input_data = Input(name='the_input', shape=self.data_shape, dtype='float32')
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (
            self.input_image_columns // (pool_size ** 2), (self.input_image_rows // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(self.alphabet_length + 1, kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels', shape=[self.maximum_number_of_characters_in_longest_text_line],
                       dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out, y_pred])

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=self.get_optimizer())

        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "simple"


if __name__ == "__main__":
    configuration = SimpleConfiguration(1900, 64, 77, 146)
    classifier = configuration.model()
    classifier.summary()
    plot_model(classifier, to_file="{0}.png".format(configuration.name()))
    print(configuration.summary())
