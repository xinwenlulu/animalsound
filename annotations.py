import numpy as np
from time_intervals import interval, find_nonoverlapping_interval, seg_available, seg_long_calls, get_duration
from numpy.random import default_rng
import matplotlib.pyplot as plt


def clip_neg_anns(target, files, starts, ends, labels, clip_length):
    ann = []
    f = 0
    calls = []
    ends = np.array(ends)
    files = np.array(files)
    starts = np.array(starts)
    for i in range(len(labels)):

        if files[i] != f or i == len(labels) - 1:
            endtime = 45
            if f == 10:
                endtime = 60
            if f == 106:
                endtime = 3292
            # find purely negative >= clip_length segments from positive files
            if len(calls) > 0:
                n = find_nonoverlapping_interval(calls, interval([0, endtime]), clip_length)
                nseg = seg_available(n, clip_length, True)
                for s, e in nseg:
                    ann.append([f, s * 1000, e * 1000, [0] * len(target)])

            # reset for each file
            f = files[i]
            calls = []

        # make negative clips from purely negative files
        if labels[i] != 'nothing':
            calls.append(interval([starts[i], ends[i]]))
        else:  # Nothing
            if ends[i] - starts[i] == clip_length:
                ann.append([files[i], float(starts[i]) * 1000, float(ends[i]) * 1000, [0] * len(target)])
            else:  # > clip_length
                # some annotation for negfiles have inconsitent timing but all refer to whole clip
                endtime = 45
                if files[i] == 10:
                    endtime = 60
                if files[i] == 106:
                    endtime = 3292
                nseg = seg_long_calls(0, endtime, clip_length, True)
                for s, e in nseg:
                    ann.append([files[i], s * 1000, e * 1000, [0] * len(target)])

    return np.array(ann, dtype=object)


# find target species present in the 5-second window
def find_species(itv, calls, labels):
    assert len(calls) == len(labels)
    species = []
    for i, call in enumerate(calls):
        o = call & itv
        # if there's overlap = the species's call present in the specified 5-second interval
        # the shortest vocalisation is 0 so no threshold here
        if o != interval():
            species.append(labels[i])
    return species


def make_label(target, species):
    # initialise zeros of target length
    label = [0] * len(target)

    for s in species:
        if s in target:
            i = target.index(s)
            label[i] = 1

    return label


def centre_calls(start, end, clip_length, filemaxendtime, random_generator=default_rng()):
    length = end - start
    if length == clip_length:
        return interval([start, end])
    else: # length < clip_length
        background = clip_length-length
        s = random_generator.uniform(start-background, start)
        if s < 0:
            s = 0
        e = s + clip_length
        if e > filemaxendtime:
            e = filemaxendtime
            s = filemaxendtime - clip_length
        return interval([s, e])


def equallabel(label1, label2):
    assert len(label1) == len(label2)
    for i1, i2 in zip(label1, label2):
        if i1 != i2:
            return False
    return True


# threshold for making a clip that centres this call
def included(call, encode_label, ann):
    for a in ann:
        #if this call is completely included in the annotation list
        if call in a[0] and equallabel(encode_label, a[1]):
            return True
        # if the label is exactly the same find amount of overlap
        # accpet if overlap is greater than 4s (requries at least 1s of unique recording)
        if equallabel(encode_label, a[1]):
            o = call & a[0]
            if o != interval() and get_duration(o) > 4:
                return True
    return False


def clip_anns(target, files, starts, ends, labels, clip_length, random_generator=default_rng()):
    ann = []
    f = 0
    ends = np.array(ends)
    files = np.array(files)
    starts = np.array(starts)
    calls = []
    calllabel = []

    for i in range(len(labels)):

        if files[i] != f or i == len(labels) - 2:

            longcalls = []
            longlabel = []
            tmpann = []

            endtime = 45
            if f == 10:
                endtime = 60
            if f == 106:
                endtime = 3292

            for j, call in enumerate(calls):

                if get_duration(call) <= clip_length:
                    # decide a 5-second time window where the call is in the middle
                    thisann = centre_calls(call[0][0], call[0][1], clip_length, endtime, random_generator)
                    species_present = find_species(thisann, calls, calllabel)
                    encode_label = make_label(target, species_present)
                    if not included(call, encode_label, tmpann):
                        tmpann.append([thisann, encode_label])
                else:  # longcalls
                    longcalls.append(interval([call[0][0], call[0][1]]))
                    longlabel.append(calllabel[j])

            # deal with overlaps
            # start from the last call in the list (deal with shorter longcalls first)
            j = len(longcalls) - 1

            while j >= 0:
                added_anns = [x[0] for x in tmpann]
                # find >0.5s segments from long calls that's not already included
                available = find_nonoverlapping_interval(added_anns, longcalls[j], 0.5)
                seg = seg_available(available, clip_length)
                for s, e in seg:
                    thisann = centre_calls(s, e, clip_length, endtime, random_generator)
                    species_present = find_species(thisann, calls, calllabel)
                    encode_label = make_label(target, species_present)
                    if not included(thisann, encode_label, tmpann):
                        tmpann.append([thisann, encode_label])
                j -= 1

            for a in tmpann:
                s = a[0][0][0]
                e = a[0][0][1]
                if e - s == clip_length:
                    ann.append([f, s * 1000, e * 1000, a[1]])
                else:
                    thisann = centre_calls(s, e, clip_length, endtime, random_generator)
                    species_present = find_species(thisann, calls, calllabel)
                    encode_label = make_label(target, species_present)
                    ann.append([f, thisann[0][0] * 1000, thisann[0][1] * 1000, encode_label])

            # reset for each file
            f = files[i]
            calls = []
            calllabel = []

        # all calls
        if labels[i] != 'nothing':
            calls.append(interval([starts[i], ends[i]]))
            calllabel.append(labels[i])

    return np.array(ann, dtype=object)


def count_label(multilabels):
    numlabels = []
    for label in multilabels:
        total = 0
        for num in label:
            if num == 1:
                total += 1
        numlabels.append(total)

    return np.array(numlabels)


def read_ann(path, nontarget_moved=False, verbose=False, plot = False):
    x = []
    y_labels = []
    for line in open(path):
        if line.strip() != "":
            # handle empty rows in file
            row = line.strip().split(" ")
            x.append(row[0:3])
            if nontarget_moved == True:
                #after removing non-target from positives
                label = ",".join(row[3:])
                label = label.strip().split(",")
            else:
                #before removing non-target from positives
                label = " ".join(row[3:])
                label = label[1:-1].strip().split(",")
            y_labels.append(list(map(int, label)))
    x = np.array(x)
    if verbose:
        y = np.array(y_labels)
        #some descriptive stats
        numlabels = count_label(y)
        proportion = sum(numlabels)/(len(y)*30)
        print('total calls: ', sum(numlabels))
        print("proportion of calls in label: ", proportion)
        print("Single Species", len(numlabels[numlabels==1]))
        print("Multiple Species", len(numlabels[numlabels>1]))
        print("Non-target Species", len(numlabels[numlabels==0]))
        print("Average", np.mean(numlabels))
        print("Maxi", max(numlabels))
    if plot:
        plot_calls(numlabels)
    return np.hstack((x, y_labels))


def plot_calls(numlabels):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Calls per Clip')
    ax1.boxplot(numlabels)


def move_nontarget_to_negatives(ann, nann, over3flist, verbose=False):
    sample_non_target = np.array(['0'] * len(over3flist))
    non_target = []
    for i, a in enumerate(ann):
        thislabel = a[3:]
        if all(thislabel == sample_non_target):
            nann = np.append(nann, np.expand_dims(a, axis=0), 0)
            non_target.append(i)
    ann = np.delete(ann, non_target, 0)
    np.savetxt('positive_multilabel.txt', ann, fmt='%s')
    np.savetxt('negatives_multilabel.txt', nann, fmt='%s')
    ann = read_ann("positive_multilabel.txt", True, verbose=verbose)
    nann = read_ann("negatives_multilabel.txt", True, verbose=verbose)
    return ann, nann


def make_ann(clip_length, over3flist, file, start, end, y_labels, move_nontarget=True,
             random_generator=default_rng(), verbose=False):
    nann = clip_neg_anns(over3flist, file, start, end, y_labels, clip_length)
    ann = clip_anns(over3flist, file, start, end, y_labels, clip_length, random_generator=random_generator)
    np.savetxt('positive_multilabel.txt', ann, fmt='%s')
    np.savetxt('negatives_multilabel.txt', nann, fmt='%s')
    ann = read_ann("positive_multilabel.txt", verbose=verbose)
    nann = read_ann("negatives_multilabel.txt", verbose=verbose)
    if move_nontarget:
        ann, nann = move_nontarget_to_negatives(ann, nann, over3flist, verbose=verbose)
    return ann, nann

