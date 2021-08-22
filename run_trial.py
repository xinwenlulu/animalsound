from configuration import f1_m, precision_m, recall_m
from baseline import baselinetrain, evaluate_on_test
from prepare_dataset import get_dataset_info
from plotevaluation import plot_training_metrics, classification_result, plot_confusion_matrix
import tensorflow as tf

TFRECORDS_FOLDER = "./tfrecords"
modelname = '3CNN'
learning_rate = 0.001
epochs = 150
patience = 30


path_list_dict, partition_size_dict = get_dataset_info(TFRECORDS_FOLDER)
print(partition_size_dict)
model, history = baselinetrain(learning_rate, f1_m, precision_m, recall_m, path_list_dict, partition_size_dict, epochs=epochs, patience=patience)

plot_training_metrics(modelname, history)

score, test_y, bool_predict = evaluate_on_test(model, path_list_dict)
print(score)

classification_result(modelname, test_y, bool_predict)
plot_confusion_matrix(modelname, test_y, bool_predict)