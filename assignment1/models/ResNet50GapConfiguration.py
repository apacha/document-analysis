from keras.applications import ResNet50
from keras.engine import Model
from keras.layers import Activation, Convolution2D, GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.utils import plot_model

from models.TrainingConfiguration import TrainingConfiguration


class ResNet50GapConfiguration(TrainingConfiguration):
    """ A network with residual modules """

    def __init__(self, width: int, height: int):
        super().__init__(data_shape=(height, width, 3))

    def classifier(self) -> Model:
        """ Returns the model of this configuration """
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.data_shape, pooling=None)
        x = base_model.output
        x = Convolution2D(8, kernel_size=(1,1), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('linear', name='output_coordinates')(x)
        model = Model(inputs=base_model.inputs, outputs=x)
        model.compile(self.get_optimizer(), loss="mean_squared_error", metrics=["mae"])

        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "res_net_50_gap"


if __name__ == "__main__":
    configuration = ResNet50GapConfiguration(1920, 1080)
    classifier = configuration.classifier()
    classifier.summary()
    plot_model(classifier, to_file="{0}.png".format(configuration.name()))
    print(configuration.summary())
