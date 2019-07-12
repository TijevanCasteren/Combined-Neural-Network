# Combined-Neural-Network
A combined neural network for recognizing spoken and written numbers, ranging from 0 to 9.

The dataset consists of five parts; spoken_test, spoken_train, written_test, written_train and match_train. The written datasets consists of MNIST images, the spoken datasets of MFCC values of spoken numbers in Arabic. The match_train dataset consists of boolean values. The value is True if the written and spoken number refer to the same number, and False if not.

This program combines a convolutional network with a recurrent neural network. After testing, an accuracy of 99.1% was achieved, with room for improvement.


