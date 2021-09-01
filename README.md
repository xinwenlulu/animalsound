# Animal Sound Classification and Retrieval

This project contains a TensorFlow input pipeline that extract multiple features from an acoustic dataset provided by University of Kent.
The pipeline proceses annotated audios into 5-second clips, extract features, logmel spectrograms, MFCCs and the raw wavegram, perform file-dependent partitioning for each clip, and serialise all the information into binary tfrecords.
It requires code in the common folder provided by Georgios Rizos
Run preprocess.py to produce the tfrecords for a multi-label classification task.

This project also contains 4 experiments that investigated the effects of data augmentation (SpecAug: experiment 1.py), an RNN layer (GRU: experiment2.py), multi-head attention (experiment3.py) and ResNet18 with and without the Squeeze-and-Excitation mechanism (SE_ResNet18.py) on retrieval of species present in each clip.


