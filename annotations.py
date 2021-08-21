from data_helper import over3flist, file, start, end, y_labels
import numpy as np
from time_intervals import interval, find_nonoverlapping_interval

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
                if f == 106:
                    print(endtime, n)
                    print(calls)
                    print(find_nonoverlapping_interval(calls, interval([0, 60]), clip_length))
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


nann = clip_neg_anns(over3flist, file, start, end, y_labels, 5)
print(nann.shape)
for x in nann:
    print(x)
#np.savetxt('negatives_multilabel.txt', nann, fmt='%s')