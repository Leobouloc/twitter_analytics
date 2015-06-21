# -*- coding: utf-8 -*-
"""
Created on Tue Apr 07 23:35:15 2015

@author: leo
"""

import pandas as pd
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import TanhLayer


def make_pybrain_ds(table, prediction_cols, to_predict, normalise = True):
    '''Takes pandas data frame and returns pybrain ds
    prediction_cols : columns used to make prediction
    to_predict : 
    '''
    print 'Creating dataset...'
    if normalise:
        table = table[prediction_cols].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    ds = SupervisedDataSet(len(prediction_cols), len(to_predict))
    for row in table.iterrows():
        # FI: row is tuple (index, Serie)
        ds.addSample(tuple(row[1][prediction_cols]), tuple(row[1][to_predict]))
    print 'Dataset created'
    return ds
    
              

def nn_predict(train, test, prediction_cols, to_predict,
               n_nodes,
               hiddenclass,
               learningrate,
               num_epochs,
               verbose = True):
                   
    ds = make_pybrain_ds(train, pour_predire_cols, to_predict)
    ds_test = make_pybrain_ds(test, pour_predire_cols, to_predict)                   

    net = buildNetwork( ds.indim, n_nodes, ds.outdim, bias = True, hiddenclass = eval(hiddenclass))
    trainer = BackpropTrainer(net, dataset=ds, learningrate= learningrate, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)   
    
    if to_predict == 'place_geny':
        train = train[train.is_place]
        
    if verbose:
        print 'XXXXXXXXXXXXXXXXXXXXXXXXXX'
        print 'Predicting :', to_predict
        print 'n_nodes_1 :', n_nodes_1
        print 'n_nodes_2 :', n_nodes_2
        print 'Layer :', hiddenclass
        print 'learningrate :', learningrate


    for epoch in range(num_epochs):
        trainer.train()
        a = pd.DataFrame(net.activateOnDataset(ds_test))
        a.columns = [to_predict + '_predict']
        a.index = test.index
        test[to_predict + '_predict'] = a[to_predict + '_predict']
        
    return (trainer, test)

# Network

#n_nodes_1 = 14 #
#n_nodes_2 = None # None if no second level
#
#hiddenclass = 'SigmoidLayer'
#learningrate = 0.02
#num_epochs = 3
#verbose = True

#test_copy = test.copy()
#train_copy = train.copy()

#for col in pour_predire_cols:
#    train[col] = (train[col] - train[col].mean()) / train[col].std()
#    test[col] = (test[col] - test[col].mean()) / test[col].std()

if __name__ == '__main__':
    print 'len pour_predire_cols', len(pour_predire_cols)
    test_1 = nn_predict(train, test, pour_predire_cols, 
                        to_predict = 'is_place',    
                        n_nodes_1 = 20, n_nodes_2 = None,
                        hiddenclass = 'SigmoidLayer',
                        learningrate = 0.05,
                        num_epochs = 5)
    
