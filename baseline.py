import tensorflow as tf
from prepare_dataset import get_test_ready
import numpy as np
from prepare_dataset import get_training_dataset, get_validation_dataset, BATCH_SIZE

checkpoint_filepath = '.'

def binarise(prediction):
    bool_predict = np.zeros(prediction.shape)
    for i, vec in enumerate(prediction):
        for j, x in enumerate(vec):
            if x < 0.5:
                bool_predict[i][j] = 0
            else:
                print("positive")
                bool_predict[i][j] = 1
    return tf.convert_to_tensor(bool_predict, dtype=tf.float32)


def baselinetrain(learning_rate, f1_m, precision_m, recall_m, path_list_dict, partition_size_dict, epochs, patience):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=[500,128, 1]),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30,'sigmoid')
    ])
    model.summary()
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    validation_steps = partition_size_dict['devel'] // BATCH_SIZE
    steps_per_epoch = partition_size_dict['train'] // BATCH_SIZE

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_f1_m',
            mode='max',
            save_best_only=True)
    ]

    history = model.fit(get_training_dataset(path_list_dict),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=my_callbacks,
                        validation_data=get_validation_dataset(path_list_dict),
                        validation_steps=validation_steps)

    return model, history


def evaluate_on_test(model, path_list_dict):
    test_x, test_y = get_test_ready(path_list_dict)
    predict_y = model.predict(test_x)
    bool_predict = binarise(predict_y)
    score = model.evaluate(x=test_x, y=test_y, verbose=1)
    return score, test_y, bool_predict