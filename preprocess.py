from numpy.random import default_rng
from annotations import make_ann
from data_helper import datawrapup
from feature_partition import make_tfrecords

seed_value = 68103
clip_length = 5
# threshold: to find species that appeared in at least 3 files
threshold = 3
verbose = True

rg = default_rng(seed_value)
clip_Folder = '../CLIPS'
data_file = '../Species_Record_Sheet_SMitchell.xlsx'


over3flist, sorted_species, file, start, end, y_labels = datawrapup(data_file, threshold=threshold, verbose=verbose)
ann, nann = make_ann(clip_length, over3flist, file, start, end, y_labels, move_nontarget=True, random_generator=rg)
make_tfrecords(ann, nann, clip_Folder, over3flist, sorted_species, rg, verbose=verbose)
