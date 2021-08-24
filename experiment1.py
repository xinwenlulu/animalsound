from configuration import f1_m, precision_m, recall_m
from models import createCNN, evaluate_on_test
from prepare_dataset import get_dataset_info
from plotevaluation import plot_training_metrics, classification_result, plot_confusion_matrix, plot_f1, figurepath
from tensorflow.keras.utils import plot_model
import numpy as np


TFRECORDS_FOLDER = "./tfrecords"
learning_rate = 0.01
epochs = 100
patience = 50
num_trials = 3
modelname = ['3CNN', 'SpecAug24', 'SpecAug40']
augmentation = [False, True, True]
timemask = [None, 24, 40]


path_list_dict, partition_size_dict = get_dataset_info(TFRECORDS_FOLDER)
print(partition_size_dict)

performance = np.empty(shape=(len(modelname)*(num_trials+1), 5))
index = 0
all_val_f1 = []

for n, a, t in zip(modelname, augmentation, timemask):

    histories = []
    average = np.zeros(shape=(5,))

    for i in range(num_trials):
        name = n + str(a) + str(t) + str(i)

        model, history = createCNN(learning_rate, f1_m, precision_m, recall_m, path_list_dict,
                                   partition_size_dict, epochs=epochs, patience=patience,
                                   filters=[32, 64, 128], activations=['relu', 'relu'], denselayers=[],
                                   augmentation=a, timemask=t)
        histories.append(history)
        score, test_y, bool_predict = evaluate_on_test(model, path_list_dict)
        classification_result(n, test_y, bool_predict)
        performance[index, ] = score
        average += score
        plot_confusion_matrix(name, test_y, bool_predict)
        index += 1

    plot_model(model, to_file=figurepath+n+'.png')
    performance[index, ] = average/num_trials
    val_f1 = plot_training_metrics(n, histories, num_trial=num_trials)
    all_val_f1.append(val_f1)
    index += 1


np.savetxt(figurepath+'Experiment1results.txt', performance, fmt='%.2f')
plot_f1(modelname, all_val_f1)



