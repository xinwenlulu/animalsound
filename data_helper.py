from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt

data_file = '../Species_Record_Sheet_SMitchell.xlsx'


def read_annotation(path):
    # Load the entire workbook.
    wb = load_workbook(path, data_only=True)
    ws = wb['Sheet1']
    all_rows = list(ws.rows)

    y_labels = []
    file = []
    start = []
    end = []
    for line in all_rows[21:]:
        file.append(line[0].value)
        start.append(line[1].value)
        end.append(line[2].value)
        label = line[3].value
        if 'Blue-eared Barbet' in label:
            label = 'Blue-eared Barbet'
        if 'Bushy-crested Hornbill' in label or 'Busy-crested Hornibll' in label:
            label = 'Bushy-crested Hornbill'
        if 'Maroon Woodpecker' in label:
            label = 'Maroon Woodpecker'
        if 'Plaintive Cuckoo' in label or label == 'Plaintive Cukcoo':
            label = 'Plaintive Cuckoo'
        if 'Rhinoceros Hornbill' in label:
            label = 'Rhinoceros Hornbill'
        if label == 'Green Iroa':
            label = 'green iora'
        if label == 'Fluff-backed Tit-babbler':
            label = 'Fluffy-backed Tit-babbler'
        if label == 'Slender-billed Crown':
            label = 'slender-billed crow'
        if label == 'Specatacled Bulbul':
            label = 'spectacled bulbul'
        if 'Orange-bellied' in label:
            label = 'Orange-bellied Flowerpecker'
        if label in ['Chestnut-backed Scimitar Babbler  ', 'Chestnut-backed Scimitar-babbler']:
            label = 'Chestnut-backed Scimitar-babbler'
        if label in ['Pied Fanrtail', 'Pied Fantail', 'Pied Fantail   ', 'Pied Fantial', 'Pied fantail']:
            label = 'Pied Fantail'
        if label in ['Black-headed  Bulbul',  'Black-headed Bulbil','Black-headed Bulbul', 'Black-headed Bululb']:
            label = 'Black-headed Bulbul'
        if label in ['Black-headed Babbler', 'Black-heaed Babbler']:
            label = 'Black-headed Babbler'
        if label == 'Olive-backed Woodpecker?':
            label = 'Olive-backed Woodpecker'
        y_labels.append(label.lower().rstrip())

    [classes, y] = np.unique(y_labels, return_inverse=True)
    f = np.unique(file)
    y = np.array(y)
    print('Number of files in annotation:', len(f))
    print('Number of species in annotation:', len(classes))
    print('Number of intances:', len(y))
    duration = []
    zerocount = 0
    for i, s, e in zip(file, start, end):
        d = e - s
        if d == 0:
            zerocount += 1
        if d >= 0:
            duration.append(d)

    meand = np.mean(duration)
    print('Average Duration: ', meand)
    print('std: ', np.std(duration))
    print('min: ', min(duration))
    print('max: ', max(duration))
    print('Number of calls with a duration of 0', zerocount)
    #fig1, ax1 = plt.subplots()
    #ax1.boxplot(duration)
    #plt.show()
    return file, start, end, y_labels, classes


def get_ann(species, files,starts, ends, labels):
    ann = []
    for i in range(len(labels)):
        if labels[i] in species:
            ann.append([files[i], starts[i], ends[i], labels[i]])
    ann = np.array(ann)
    return ann


def get_short_ann(files,starts, ends, labels, call_threshold):
    ann = []
    for i in range(len(labels)):
        if ends[i] - starts[i] < call_threshold:
            ann.append([files[i], starts[i], ends[i], labels[i]])
    ann = np.array(ann)
    return ann


def get_species(y_labels, count_threshold):
    classes, counts = np.unique(y_labels, return_counts=True)
    result = []
    negative = []
    for cls,count in zip(classes, counts):
        if count >= count_threshold and cls != 'nothing' and cls != 'unknown':
            result.append(cls)
        elif cls == 'nothing':
            negative.append(cls)
    # ignore Unkown
    print('Species with more than 3 annotation: ', len(result))
    return result, negative


def find_files(targetlabels, alllabels, files):
    targetfiles = []
    for i in range(len(alllabels)):
        if alllabels[i] in targetlabels and files[i] not in targetfiles:
            targetfiles.append(files[i])
    return targetfiles


def get_species_multiple_files(species, file_threshold, y_labels, file):
    resultdic = {}
    resultflist = []
    occurrence = {}
    for c in species:
        val = []
        for cls, fi in zip(y_labels,file):
            if cls == c:
                val.append(fi)
        f = np.unique(val)
        if len(f) >= file_threshold:
            resultdic[c] = val
            occurrence[c] = len(val)
            resultflist.append(c)
    return resultdic, resultflist, occurrence

def plot_occurrence(sorted_dic):
    plt.figure(figsize=(15,6))
    plt.bar(range(len(sorted_dic)), list(sorted_dic.values()), align='center')
    plt.xticks(range(len(sorted_dic)), list(sorted_dic.keys()), rotation=90)
    plt.show()



file, start, end, y_labels, classes = read_annotation(data_file)
over3, negative = get_species(y_labels, 3)
fileover3 = find_files(over3,y_labels,file)
filenegative = find_files(negative,y_labels,file)
print('Files that contain species with more than 3 annotation', len(fileover3))
print('Files that contain Negative files', len(filenegative))
over3dic, over3flist, occurrence = get_species_multiple_files(over3, 3, y_labels, file)
print('Species that appear in at least 3 files: ', len(over3flist))
sorted_dic = {k: v for k, v in sorted(occurrence.items(), key=lambda item: item[1])}
sorted_species = sorted_dic.keys()
#plot_occurrence(sorted_dic)