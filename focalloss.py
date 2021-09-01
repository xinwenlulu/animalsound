from metrics import f1_m, precision_m, recall_m
from models import CNNwithFocalLoss, evaluate_on_test
from prepare_dataset import get_dataset_info
from plotevaluation import plot_training_metrics, classification_result, plot_confusion_matrix, plot_f1, figurepath
import numpy as np
import tensorflow as tf

TFRECORDS_FOLDER = "./tfrecords"
learning_rate = 0.01
epochs = 100
patience = 50
num_trials = 3
modelname = 'focalLoss'
augmentation = True
timemask = 24

path_list_dict, partition_size_dict = get_dataset_info(TFRECORDS_FOLDER)
print(partition_size_dict)

for i in range(num_trials):
    name = modelname+str(i)
    model, history = CNNwithFocalLoss(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
                                   partition_size_dict, epochs=epochs, patience=patience,
                                   filters=[32, 64, 128], activations=['relu', 'relu'], denselayers=[],
                                   augmentation=augmentation, timemask=timemask)

    score, test_y, bool_predict = evaluate_on_test(model, path_list_dict, focal_loss=True)

    classification_result(modelname, tf.reshape(test_y, [467, 30]), tf.reshape(bool_predict, [467, 30]))
    plot_confusion_matrix(name, tf.reshape(test_y, [467, 30]), tf.reshape(bool_predict, [467, 30]))
    val_f1 = plot_training_metrics(name, history)
    np.savetxt(figurepath + name + 'valF1.txt', val_f1, fmt='%.2f')