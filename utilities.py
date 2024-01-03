import h5py
import numpy as np


def load_data():
    train_dataset = h5py.File('datasets/train_2.h5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_2.h5', "r")
    X_test = np.array(test_dataset["X_train"][:]) # your train set features
    y_test = np.array(test_dataset["Y_train"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test