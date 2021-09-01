import tensorflow as tf
from prepare_dataset import get_test_ready
from prepare_dataset import get_training_dataset, get_validation_dataset, BATCH_SIZE
from dataAugmentation import DataGenerator
from metrics import my_focal_loss
from keras import backend as K
from common.models.embedding_pooling import AttentionGlobalPooling
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  \
    BatchNormalization, Add, multiply, MultiHeadAttention, MaxPooling2D, GRU, Reshape, \
    GlobalAveragePooling1D, Input
from keras.models import Sequential
from keras.models import Model

checkpoint_filepath = '.'
figurepath = './figures/'


def binarise(prediction):
    y_pred_f = K.cast(K.greater(prediction, 0.5), 'float32')
    return y_pred_f


def initialiseModel(learning_rate, model, f1_m, precision_m, recall_m,
                    partition_size_dict, path_list_dict, patience, epochs,
                    augmentation=None, timemask=24):

    model.summary()

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    validation_steps = partition_size_dict['devel'] // BATCH_SIZE
    steps_per_epoch = partition_size_dict['train'] // BATCH_SIZE

    my_callbacks = [
        EarlyStopping(patience=patience),
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_f1_m',
            mode='max',
            save_best_only=True)
    ]

    x_train, y_train = get_training_dataset(path_list_dict)
    if augmentation:
        augGenerator = DataGenerator(x_train, y_train, batch_size=64, timemask=timemask)
        history = model.fit(augGenerator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=get_validation_dataset(path_list_dict),
                            validation_steps=validation_steps)
        model.load_weights(checkpoint_filepath)
        return model, history

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=my_callbacks,
                        validation_data=get_validation_dataset(path_list_dict),
                        validation_steps=validation_steps)
    model.load_weights(checkpoint_filepath)
    return model, history


def addAttention(model):
    attention = AttentionGlobalPooling(number_of_heads=4, use_temporal_std=False, pool_heads="attention",
                           auto_pooling="no_auto", number_of_features=992, sequence_length=128,
                           use_auto_array=False, outputs_list=(30,))
    model.add(attention)
    return model


def addCNNlayers(model, filters, denselayers, activations, RNNlayers=None):
    model.add(tf.keras.Input(shape=[500, 128, 1]))
    for i in range(len(filters)):
        model.add(Conv2D(filters[i], kernel_size=(3, 3), padding='same', activation=activations[0]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    if RNNlayers is not None:
        model.add(Reshape((62, 2048)))
        for i in range(len(RNNlayers)):
            model.add(GRU(RNNlayers[i], return_sequences=True))
        model.add(GlobalAveragePooling1D())
    else:
        model.add(GlobalAveragePooling2D())

    if len(denselayers) > 0:
        for dense in denselayers:
            model.add(Dense(dense, activations[1]))
    return model


def createCNN(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
              partition_size_dict, epochs, patience, filters=[32, 64, 128],
              activations=['relu', 'relu'], denselayers=[], augmentation=False, timemask=24):

    model = tf.keras.Sequential()
    model = addCNNlayers(model, filters, denselayers, activations)
    model.add(Dense(30, 'sigmoid'))

    return initialiseModel(learning_rate, model, f1_m, precision_m, recall_m,
                           partition_size_dict, path_list_dict, patience, epochs,
                           augmentation=augmentation, timemask=timemask)


def CNNwithFocalLoss(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
              partition_size_dict, epochs, patience, filters=[32, 64, 128],
              activations=['relu', 'relu'], denselayers=[], augmentation=False, timemask=24):

    model = Sequential()
    model = addCNNlayers(model, filters, denselayers, activations)
    model.add(Dense(30, 'sigmoid'))
    model.add(Reshape((1, 30), input_shape=(30,)))

    model.summary()

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=my_focal_loss,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    validation_steps = partition_size_dict['devel'] // BATCH_SIZE
    steps_per_epoch = partition_size_dict['train'] // BATCH_SIZE

    my_callbacks = [
        EarlyStopping(patience=patience),
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_f1_m',
            mode='max',
            save_best_only=True)
    ]

    x_train, y_train = get_training_dataset(path_list_dict, focal_loss=True)
    if augmentation:
        augGenerator = DataGenerator(x_train, y_train, batch_size=64, timemask=timemask, focal_loss=True)
        history = model.fit(augGenerator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=get_validation_dataset(path_list_dict,focal_loss=True),
                            validation_steps=validation_steps)
        model.load_weights(checkpoint_filepath)
        return model, history

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=my_callbacks,
                        validation_data=get_validation_dataset(path_list_dict),
                        validation_steps=validation_steps)
    model.load_weights(checkpoint_filepath)
    return model, history



def createCRNN(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
              partition_size_dict, epochs, patience, filters=[32, 64, 128],
              activations=['relu', 'relu'], denselayers=[], RNNlayers=[128],
               augmentation=False, timemask=24):

    model = Sequential()
    model = addCNNlayers(model, filters, denselayers, activations, RNNlayers=RNNlayers)
    model.add(Dense(30, 'sigmoid'))

    return initialiseModel(learning_rate, model, f1_m, precision_m, recall_m,
                           partition_size_dict, path_list_dict, patience, epochs,
                           augmentation=augmentation, timemask=timemask)


def createAttentionModel(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
              partition_size_dict, epochs, patience, filters=[32, 64, 128],
              activations=['relu', 'relu'], augmentation=False, timemask=24):

    input = Input(shape=[500, 128, 1])
    cnn1 = Conv2D(filters[0], kernel_size=(3, 3), padding='same', activation=activations[0])(input)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    cnn2 = Conv2D(filters[1], kernel_size=(3, 3), padding='same', activation=activations[0])(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(cnn2)
    cnn3 = Conv2D(filters[2], kernel_size=(3, 3), padding='same', activation=activations[0])(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(cnn3)
    #reshape = Reshape((62, 16*128))(maxpool3)
    attention = MultiHeadAttention(num_heads=4, key_dim=2, attention_axes=1)(maxpool3, maxpool3)
    #attention = AttentionGlobalPooling(number_of_heads=4, use_temporal_std=False, pool_heads="attention",
                                       #auto_pooling="no_auto", number_of_features=992, sequence_length=128,
                                       #use_auto_array=False, outputs_list=(30,))(reshape)
    avgpool = GlobalAveragePooling2D()(attention)
    output = Dense(30, 'sigmoid')(avgpool)
    model = Model(inputs=input, outputs=output)

    return initialiseModel(learning_rate, model, f1_m, precision_m, recall_m,
                           partition_size_dict, path_list_dict, patience, epochs,
                           augmentation=augmentation, timemask=timemask)


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False, se=False, ratio=16):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self._se = se
        self._ratio = ratio
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

        if self._se:
            self.squeeze = GlobalAveragePooling2D()
            self.fc_1 = Dense(self.__channels // self._ratio, activation='relu')
            self.fc_2 = Dense(self.__channels, activation='sigmoid')

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)

        if self._se:
            squeeze = self.squeeze(out)
            excitation = self.fc_1(squeeze)
            excitation = self.fc_2(excitation)
            out = multiply([out, excitation])

        return out


class SEResNet18(Model):

    def __init__(self, num_classes, se=False, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64, se=se)
        self.res_1_2 = ResnetBlock(64, se=se)
        self.res_2_1 = ResnetBlock(128, down_sample=True, se=se)
        self.res_2_2 = ResnetBlock(128, se=se)
        self.res_3_1 = ResnetBlock(256, down_sample=True, se=se)
        self.res_3_2 = ResnetBlock(256, se=se)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
                          self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out


def createSEResNet18(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
              partition_size_dict, epochs, patience, augmentation=False, timemask=24, se=False):
    model = SEResNet18(30, se=se)
    model.build(input_shape=(None, 500, 128, 1))
    return initialiseModel(learning_rate, model, f1_m, precision_m, recall_m,
                           partition_size_dict, path_list_dict, patience, epochs,
                           augmentation=augmentation, timemask=timemask)



def evaluate_on_test(model, path_list_dict, focal_loss=False):
    test_x, test_y = get_test_ready(path_list_dict, focal_loss=focal_loss)
    predict_y = model.predict(test_x)
    bool_predict = binarise(predict_y)
    score = model.evaluate(x=test_x, y=test_y, verbose=1)
    return score, test_y, bool_predict
