from bsa.bsa import *
from bsa.utils import get_accuracy
from bsa.utils import train_test_split
import scipy.io as scio
import time
__author__ = 'tangy'

def main():
    mat = scio.loadmat("data.mat")["TrainingData"]
    num_samples,num_features = mat.shape
    num_features -= 1
    X = mat[:,:num_features]
    y = mat[:,num_features]
    X_train,X_test,y_train,y_test = train_test_split(X,y,divRatio=0.2)
    start = time.time()
    classifier = BSANN_OVA(hidden_layer_size=35,verbose=True,maxiters=10,n_jobs=-1)
    classifier.fit(X_train,y_train)
    print "Training Accuracy : " + str(get_accuracy(y_train,classifier.predict(X_train))*100)
    print "Testing Accuracy : " + str(get_accuracy(y_test,classifier.predict(X_test))*100)
    print "Timing"
    print str(time.time() - start)

if __name__=="__main__":
    main()
