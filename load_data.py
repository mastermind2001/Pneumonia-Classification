import numpy as np
import h5py

def shuffle_data(X, Y):
    permutation = list(np.random.permutation(X.shape[0]))
    X = X[permutation,:,:,:]
    Y = Y[permutation,:]

    return X, Y

def load_dataset():
    #loading training data
    with h5py.File('train.h5', 'r') as f1:
      X_train = np.array(f1['train']['train_X'][:])
      Y_train = np.array(f1['train']['train_Y'][:])

    #loading validation data
    with h5py.File('val.h5', 'r') as f2:
      X_val = np.array(f2['validate']['val_X'][:])
      Y_val = np.array(f2['validate']['val_Y'][:])

    #loading test data
    with h5py.File('test.h5', 'r') as f3:
      X_test = np.array(f3['test']['test_X'][:])
      Y_test = np.array(f3['test']['test_Y'][:])

    #shuffle training data
    X_train, Y_train = shuffle_data(X_train, Y_train)

    #shuffle validation data
    X_val, Y_val = shuffle_data(X_val, Y_val)

    #shuffle test data
    X_test, Y_test = shuffle_data(X_test, Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

