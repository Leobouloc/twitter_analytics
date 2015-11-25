# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 03:32:05 2015

@author: leo

Trying different stuff here : 
One part is just loading the data.
Another part is creating a co-existance frequency table
Another part is implementing a "word2vec-like" thing to assign vectors to politicians  

"""

import numpy as np
from math import sqrt
import pandas as pd
from os import listdir
from os.path import isfile, join
import itertools

import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

import pandas_pybrain
import pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import TanhLayer


    

def create_ids_tab(path_ids_group, group_name):
    '''Creates table of ids in record format for group group_name'''
    print 'Creating ids_tab'
    files = [ f for f in listdir(path_ids_group) if isfile(join(path_ids_group,f)) ]    
    ids_tab_rec = pd.DataFrame(columns = [group_name, 'id'])
    all_ids = dict()
    for file in files:
        f = open(join(path_ids_group, file), 'r')
        ids = f.read().split('\n')
        f.close()
        local_tab = pd.DataFrame()
        local_tab['id'] = ids[:-1]
        local_tab[group_name] = file.replace('_ids.txt', '')
        ids_tab_rec = ids_tab_rec.append(local_tab)
        all_ids[file.replace('_ids.txt', '')] = ids
    ids_tab_rec['True'] = True
    ids_tab = pd.pivot_table(ids_tab_rec, values = 'True', columns = group_name, index = 'id').fillna(False)
    print 'Ids_tab created'
    return (all_ids, ids_tab_rec, ids_tab)



def _make_train_set(row):
    '''Assumes all columns in row are words in vocabulary'''
    new_columns = list(row.index) + ['context_' + col for col in row.index]
    return_tab = pd.DataFrame(columns = new_columns)
    list_row = list(row)
    len_row = len(row)
    for i, val in enumerate(list_row):
        if val:
            new_list = [False]*i + [True] + [False]*(len_row - i - 1) + list_row[:i] + [False] + list_row[i+1:]
            return_tab = return_tab.append(pd.DataFrame([new_list], columns = new_columns))
    return return_tab

path = '.'
path_ids = join(path, 'data', 'ids')
path_ids_news_fr = join(path_ids, 'news_fr')
path_ids_politiques_fr = join(path_ids, 'politiques_fr')
print 'here3'   

(all_ids, ids_tab_rec, ids_tab) =  create_ids_tab(path_ids_politiques_fr, 'politique')



# Perform PCA
from sklearn.decomposition import PCA
import random
from mpl_toolkits.mplot3d import Axes3D

ids_tab = ids_tab[ids_tab.sum(axis = 1) >= 2]
index = random.sample(ids_tab.index, min(1000000, len(ids_tab))) # Max 1M
ids_tab = ids_tab.loc[index,:]


pca = PCA(n_components=5)
pca.fit(ids_tab)

eigenvectors = pd.DataFrame(columns = ids_tab.columns)
for i, row in enumerate(pca.components_):
    eigenvectors.loc[i, :] = row
    
# Make plot according to top eigenvectors
a = eigenvectors.loc[0, :] # 
b = eigenvectors.loc[1, :] # 
c = eigenvectors.loc[2, :] #
d = eigenvectors.loc[3, :] #

x = (ids_tab*a).sum(axis = 1)
y = (ids_tab*b).sum(axis = 1)
z = (ids_tab*c).sum(axis = 1)
g = (ids_tab*d).sum(axis = 1)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
index = random.sample(ids_tab.index, 30000)

#ax.scatter(x[index], y[index], z[index], c='r', marker='x') # 3D
plt.scatter(y[index], z[index], c='r', alpha=0.3) # 2D
plt.show()
assert False

ids_tab_for_input = ids_tab[ids_tab.sum(axis = 1) >= 2]
# Make actual word2vec (take out additional value ie 'johnny has toys' --> johnny predicts has and toys)
print 'here2'
#ids_tab = ids_tab[ids_tab.sum(axis = 1) > 1]
input_table = pd.DataFrame()
count = 0
import datetime
tic = datetime.datetime.now()
for ind, row in ids_tab.iloc[:20000].iterrows():
    if count%100 == 0:
        print 'Did', count, 'in', datetime.datetime.now() - tic
        tic = datetime.datetime.now()
    count += 1
    input_table = input_table.append(_make_train_set(row))
input_table.index = range(len(input_table))
input_table = input_table[input_table.sum(axis = 1) > 1]
print 'here1'

train_type = 'word2vec_ish' # 'autoencoder' or ''word2vec-ish'
pred_cols = ids_tab.columns
context_cols = ['context_' + col for col in pred_cols]
if train_type == 'autoencoder':
    ds = pandas_pybrain.make_pybrain_ds(ids_tab.iloc[:2000], pred_cols, pred_cols, normalise = False)             
elif train_type == 'word2vec_ish':
    ds = pandas_pybrain.make_pybrain_ds(input_table, pred_cols, context_cols, normalise = False)             

print 'here'

n_nodes = 10
hiddenclass = 'SigmoidLayer'
learningrate = 0.05

# Build Network
net = buildNetwork(ds.indim, n_nodes, ds.outdim, bias = True, hiddenclass = eval(hiddenclass))
trainer = BackpropTrainer(net, dataset=ds, learningrate= learningrate, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)   
    
# Train Network
for epoch in range(10):
    print epoch
    trainer.train()

vec_table = pd.DataFrame(index = pred_cols, columns = range(n_nodes))
for i, col in enumerate(pred_cols):
    inpt = [False]*i + [True] + [False]*(len(pred_cols) - 1 - i)
    net.activate(inpt)
    vec = net['hidden0'].outputbuffer[net['hidden0'].offset]
    vec_table.loc[col] = list(vec)

distance_tab = pd.DataFrame(columns = pred_cols, index = pred_cols)
for col in pred_cols:
    for index in pred_cols:
        distance_tab.loc[index, col] = cosine(vec_table.loc[index], vec_table.loc[col])

import matplotlib.pyplot as plt

plt.pcolor(distance_tab)
plt.yticks(np.arange(0.5, len(distance_tab.index), 1), distance_tab.index)
plt.xticks(np.arange(0.5, len(distance_tab.columns), 1), distance_tab.columns)
plt.show()


assert False

num_following = ids_tab.groupby('id').size()
num_following.name = 'num_following'
weight = 1/num_following
weight.name = 'weight'
#ids_tab = ids_tab.join(num_following, on = 'id')

#assert False


#for politique in ['fhollande', 'nicolassarkozy']:
#    f = open(join(path_ids_politiques_fr, politique + '_ids.txt'))
#    list_ids_tab_politique = f.read().split('\n')
#    ids_tab_politique = pd.DataFrame()
#    ids_tab_politique['id'] = list_ids_tab_politique
#    ids_tab_politique[politique] = True
#    f.close()
#    
#    ids_tab = ids_tab.merge(ids_tab_politique, on = 'id', how = 'left')
#    ids_tab[politique] = ids_tab[politique].fillna(False)
#
#assert False
#
#b = a.groupby('artist')[news].mean()
#b.sort()
#for i in range(len(b)):
#    print b.index[i], '  >>  ', b.iloc[i]
#
#
#d = a.groupby('id').size()
#e = pd.DataFrame()
#e['count'] = d
#a = a.merge(e, left_on = 'id', right_index = True)    

affinity_mat = pd.DataFrame()
grp = ids_tab.groupby('politique')
for (key_x, key_y) in itertools.combinations(ids_tab.politique.unique(), 2):
    print (key_x, key_y)
    x = grp.get_group(key_x)
    y = grp.get_group(key_y)
    z = pd.merge(x, y, how = 'inner', on = 'id')
    z = z.join(weight, on = 'id')
    val = z.weight.sum()
    affinity_mat.loc[key_x, key_y] = val
    affinity_mat.loc[key_y, key_x] = val

#for (key_x, key_y) in itertools.combinations(ids_tab.politique.unique(), 2):

for key_x in ids_tab.politique.unique():
    print key_x
    x = grp.get_group(key_x)
    z = x.join(weight, on = 'id')
    affinity_mat.loc[key_x, key_x] = z.weight.sum()
    
    
print affinity_mat
assert False



divider_mat = pd.DataFrame(columns = affinity_mat.columns, index = affinity_mat.index)
for x in divider_mat.index:
    for y in divider_mat.columns:
        divider_mat.loc[x, y] = sqrt(affinity_mat.loc[y, y]**2)# + affinity_mat.loc[x, x]**2 #, affinity_mat.loc[y, y])
        
a = affinity_mat / divider_mat


columns = ['jlmelenchon', 'cecileduflot', u'evajoly', 'fleurpellerin', 'eelv','benoithamon', 'montebourg', 'chtaubira', 'anne_hidalgo', 'najatvb', 'harlemdesir','martineaubry', 'fhollande', 'royalsegolene', 'manuelvalls', 'partisocialiste', 'bayrou', 'modem', u'datirachida', u'alainjuppe', u'francoisfillon', u'jf_cope', u'nk_m', u'christineboutin', u'bruno_lemaire', 'vpecresse','nadine__morano',u'nicolassarkozy', 'ump', u'mlp_officiel', u'fn_officiel']
a = a.loc[columns, columns]

for i in range(len(a)):
    a.iloc[i, i] = np.nan
b = (a + a.T)/2
b = (b.T + b)/2
#for x in range(len(b)):
#    for y in range(len(b)):
#        if x > y:
#            b.iloc[x, y] = ''
            
#print a


#plt.pcolor(a)
#plt.xticks(range(len(a)), list(a.columns), rotation = 45)
#plt.yticks(range(len(a)), list(a.columns))
#plt.show()
#
#def sorted(s, num):
#    tmp = s.order(ascending=False)[:num]
#    tmp.index = range(num)
#    return tmp

for y in range(len(a)):
    print b.index[y], '   ',b.apply(lambda x: x.argmax()).iloc[y]
    
from sklearn.cluster import SpectralClustering

clustering = SpectralClustering()