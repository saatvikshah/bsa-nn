#BSA and BSA-NN

##Description
BSANN is a complex stochastic search based Evolutionary Algorithm which smartly backtracks from new to old populations during its evolution to reduce error caused from entering a non-ideal search space. It is based on a multilayer Neural Network Architecture. It has shown excellent results in the domain of EEG pattern detection beating the results of best available algorithms(including winner of the BCI competition) in the field of Motor Imagery BCI. Given is a python library implementation of both the original BSA as well as BSA-NN supporting parallel processing

##Installation
    $ python setup.py install
All dependencies will automatically be installed. In case of any error you can use

    $ pip install -r requirements.txt

##Usage
###BSA
    from bsa.bsa import BSA
    ...
    clf = BSA(maxiters=300,n_jobs=-1)   
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
###BSA-NN 
    from bsa.bsa import BSANN_OVA
    ...
    clf = BSANN_OVA(hidden_layer_size=20,maxiters=300,n_jobs=-1)   
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)

##Additional Notes
1. Currently only classification is supported
2. Docstrings/More Examples Pending