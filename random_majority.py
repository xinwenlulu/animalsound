from metrics import f1_m, precision_m, recall_m
from models import createRandomMajority, evaluate_on_test
from prepare_dataset import get_dataset_info
from plotevaluation import classification_result
import numpy as np

TFRECORDS_FOLDER = "./tfrecords"
modelname = ['Random', 'Majority']
learning_rate = 0.01

path_list_dict, partition_size_dict = get_dataset_info(TFRECORDS_FOLDER)
print(partition_size_dict)
index = 0
for name in modelname:
    model = createRandomMajority(learning_rate, f1_m, precision_m, recall_m,type=name)
    score, test_y, bool_predict = evaluate_on_test(model, path_list_dict)
    classification_result(name, test_y, bool_predict)