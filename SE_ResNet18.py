from metrics import f1_m, precision_m, recall_m
from models import createSEResNet18, evaluate_on_test
from prepare_dataset import get_dataset_info
from plotevaluation import plot_training_metrics, classification_result, plot_confusion_matrix, figurepath
from tensorflow.keras.utils import plot_model
import numpy as np

TFRECORDS_FOLDER = "./tfrecords"
learning_rate = 0.01
epochs = 100
patience = 50
num_trials = 3
modelname = ['ResNet18', 'SEResNet18']
augmentation = True
timemask = 24
se = [False, True]

path_list_dict, partition_size_dict = get_dataset_info(TFRECORDS_FOLDER)
print(partition_size_dict)
for n, s in zip(modelname, se):
    name = n
    for i in range(num_trials):
        name = n+str(i)
        model, history = createSEResNet18(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
                                          partition_size_dict, epochs=epochs, patience=patience,
                                          augmentation=augmentation, timemask=timemask, se=s)
        score, test_y, bool_predict = evaluate_on_test(model, path_list_dict)
        classification_result(name, test_y, bool_predict)
        val_f1 = plot_training_metrics(name, history)
        np.savetxt(figurepath + name + 'valF1.txt', val_f1, fmt='%.2f')
        plot_confusion_matrix(name, test_y, bool_predict)

plot_model(model, to_file=figurepath+modelname+'.png')