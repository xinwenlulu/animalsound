import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns


figurepath = './figures/'


target_names = ['ashy tailorbird', 'banded bay cuckoo', 'black-capped babbler', 'black-headed bulbul', \
                'black-naped monarch', 'blue-eared barbet', 'bold-striped tit-babbler', 'bornean gibbon', \
                'brown fulvetta', 'buff-vented bulbul', 'bushy-crested hornbill', \
                'chestnut-backed scimitar-babbler', 'chestnut-rumped babbler', 'chestnut-winged babbler',\
                'dark-necked tailorbird', 'ferruginous babbler','fluffy-backed tit-babbler', 'grey-headed babbler',\
                'little spiderhunter', 'pied fantail', 'plaintive cuckoo', 'rhinoceros hornbill', \
                'rufous-fronted babbler', 'rufous-tailed shama', 'rufous-tailed tailorbird', 'short-tailed babbler',\
                'slender-billed crow', 'sooty-capped babbler', 'spectacled bulbul', 'yellow-vented bulbul']


def plot_training_metrics(modelname, histories, num_trial):
    train_metrics = ['loss', 'accuracy', 'f1_m', 'precision_m', 'recall_m']
    val_metrics = ['val_loss', 'val_accuracy', 'val_f1_m', 'val_precision_m', 'val_recall_m']

    fig, axs = plt.subplots(5)
    index = 1
    fig.set_size_inches(16, 10)
    for t, v in zip(train_metrics, val_metrics):
        plt.subplot(2, 3, index)
        totalt = np.zeros(shape=np.max([len(x.history[t]) for x in histories], 0))
        totalv = np.zeros(shape=np.max([len(x.history[v]) for x in histories], 0))
        for i in range(num_trial):
            totalt[0:len(histories[i].history[t])] += np.array(histories[i].history[t])
            totalv[0:len(histories[i].history[v])] += np.array(histories[i].history[v])
        plt.plot(totalt/num_trial)
        plt.plot(totalv/num_trial)
        if v == 'val_f1_m':
            val_f1 = totalv/num_trial
        plt.title('model ' + t)
        plt.ylabel(t)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        index += 1
    plt.savefig(figurepath+modelname + '.png')
    return val_f1



def classification_result(modelname, test_y, bool_predict):
    report = classification_report(test_y, bool_predict, target_names=target_names)
    print("Classification report: \n", report)
    f = open(figurepath+modelname+'report.txt', "a")
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
    plt.savefig(figurepath+modelname+'cm.png')


def plot_f1(modelname, all_val_f1):
    fig, axs = plt.subplots()
    fig.set_size_inches(12, 8)
    for m in all_val_f1:
        plt.plot(m)
    plt.title('Macro F1 on Validation set')
    plt.ylabel("F1")
    plt.xlabel('epoch')
    plt.legend(modelname, loc='upper left')
    plt.savefig(figurepath + 'F1.png')