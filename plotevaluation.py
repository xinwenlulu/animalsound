import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns


target_names = ['ashy tailorbird', 'banded bay cuckoo', 'black-capped babbler', 'black-headed bulbul', \
                'black-naped monarch', 'blue-eared barbet', 'bold-striped tit-babbler', 'bornean gibbon', \
                'brown fulvetta', 'buff-vented bulbul', 'bushy-crested hornbill', \
                'chestnut-backed scimitar-babbler', 'chestnut-rumped babbler', 'chestnut-winged babbler',\
                'dark-necked tailorbird', 'ferruginous babbler','fluffy-backed tit-babbler', 'grey-headed babbler',\
                'little spiderhunter', 'pied fantail', 'plaintive cuckoo', 'rhinoceros hornbill', \
                'rufous-fronted babbler', 'rufous-tailed shama', 'rufous-tailed tailorbird', 'short-tailed babbler',\
                'slender-billed crow', 'sooty-capped babbler', 'spectacled bulbul', 'yellow-vented bulbul']



def plot_training_metrics(modelname, history):
    train_metrics = ['loss', 'accuracy', 'f1_m', 'precision_m', 'recall_m']
    val_metrics = ['val_loss', 'val_accuracy', 'val_f1_m', 'val_precision_m', 'val_recall_m']

    fig, axs = plt.subplots(5)
    index = 1
    fig.set_size_inches(16, 10)
    for t, v in zip(train_metrics, val_metrics):
        plt.subplot(2, 3, index)
        plt.plot(history.history[t])
        plt.plot(history.history[v])
        plt.title('model ' + t)
        plt.ylabel(t)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        index += 1
    plt.savefig(modelname + '.png')



def classification_result(modelname, test_y, bool_predict):
    report = classification_report(test_y, bool_predict, target_names=target_names)
    print("Classification report: \n", report)
    f = open(modelname+'report.txt', "a")
    f.write(report)
    f.close()
    print("F1 micro averaging:", (f1_score(test_y, bool_predict, average='micro')))
    print("ROC: ", (roc_auc_score(test_y, bool_predict)))


def plot_confusion_matrix(modelname, test_y, bool_predict):
    cm = multilabel_confusion_matrix(test_y, bool_predict)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    fig, axs = plt.subplots(30)
    fig.set_size_inches(14, 16)
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        group_counts = ["{0:0.0f}".format(value) for value in
                        cm[i].flatten()]
        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_names, group_counts)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cm[i], annot=labels, fmt='', cmap='Blues')
        axs[-1].axis('off')
        plt.title(target_names[i], fontsize=13)

    fig.tight_layout(h_pad=1, w_pad=0.5)
    plt.savefig(modelname+'cm.png')

