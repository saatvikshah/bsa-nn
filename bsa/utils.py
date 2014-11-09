import numpy as np
__author__ = 'tangy'

def get_accuracy(ytrue,ypred):
    return np.mean(ytrue == ypred)

def feature_normalize(X):
    """Normalizes data in case of large variations between sets of weights"""
    Xmean = np.zeros(X.shape[1])
    Xstd = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        Xmean[i] = np.mean(X[i, :])
        Xstd[i] = np.std(X[i, :])
        X[:, i] = (X[:, i] - Xmean[i]) / Xstd[i]
    return X, Xmean, Xstd

def train_test_split(X, y, inputShuffle=False, divRatio=0.75):
    """Prepares Training and Testing Sets from given Data"""
    num_samples,num_features = X.shape
    Xy = zip(X,y)
    if (inputShuffle):
        np.random.shuffle(Xy)
    train_indices = range(int(num_samples * divRatio))
    test_indices = range(int(num_samples * divRatio) + 1,num_samples)
    Xtrain,ytrain = (np.array(l) for l in zip(*[Xy[i] for i in train_indices]))
    Xtest,ytest = (np.array(l) for l in zip(*[Xy[i] for i in test_indices]))
    return Xtrain,Xtest,ytrain,ytest