from keras import Input
from keras.applications import ResNet50, resnet50
from keras.engine import Model
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Conv2D, Add, \
    UpSampling2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.utils import plot_model, get_file

from models.TrainingConfiguration import TrainingConfiguration

# See https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
               use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
               use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                        '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layres
    """

    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = 5 # resnet_50
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


class ResNet50PyramidConfiguration(TrainingConfiguration):
    """ A network with residual modules """

    def __init__(self, width: int, height: int):
        super().__init__(data_shape=(height, width, 3))

    def classifier(self) -> Model:
        """ Returns the model of this configuration """
        input_image = Input(shape=self.data_shape)
        [C1, C2, C3, C4, C5] = resnet_graph(input_image)

        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model = Model(inputs=input_image, outputs=C5)
        model.load_weights(weights_path)

        P5 = Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = Add(name="fpn_p4add")([
            UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = Add(name="fpn_p3add")([
            UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = Add(name="fpn_p2add")([
            UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P2 = Conv2D(8, kernel_size=(1, 1), padding='same')(P2)
        P2 = GlobalAveragePooling2D()(P2)

        P3 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P3 = Conv2D(8, kernel_size=(1, 1), padding='same')(P3)
        P3 = GlobalAveragePooling2D()(P3)

        P4 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P4 = Conv2D(8, kernel_size=(1, 1), padding='same')(P4)
        P4 = GlobalAveragePooling2D()(P4)

        P5 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        P5 = Conv2D(8, kernel_size=(1, 1), padding='same')(P5)
        P5 = GlobalAveragePooling2D()(P5)

        x = Add()([P2, P3, P4, P5])
        x = Activation('linear', name='output_coordinates')(x)
        model = Model(inputs=input_image, outputs=x)
        model.compile(self.get_optimizer(), loss="mean_squared_error", metrics=["mae"])

        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "res_net_50_pyramid"


if __name__ == "__main__":
    configuration = ResNet50PyramidConfiguration(800, 448)
    classifier = configuration.classifier()
    classifier.summary()
    plot_model(classifier, to_file="{0}.png".format(configuration.name()))
    print(configuration.summary())
