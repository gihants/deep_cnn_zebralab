# deep_cnn_zebralab
Deep Convolutional Neural network (Google's Inception-v3) to detect and classify erroneous tracks created by Zebralab due to software malfunction

## Prerequisites
[Tensorflow] (https://www.tensorflow.org/) is required with Python 3.6 or above.


## Training
Training images are located under the folder [training_dataset] under two folders for the classes 'include' and exclude'.

Hyper-parameters related to the training process can be modified within the file [train.sh].

To run the training:
```bash
./train.sh
```
Tensorboard can be initiated using:
```bash
tensorboard --logdir /tmp/retrain_logs
```

## Clasify new images
New images to be classified are to be placed within the folder [to_clasify].

Classification can be initiated by:
```bash
python classify_files.py
```

Depending on the confidence score of an image to contain correct tracks only, it will be placed within a folder in the range of [exclude_90_100],......, [exclude_50_60], [include_50_60],......., [include_90_100].
