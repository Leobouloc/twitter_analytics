# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 03:32:05 2015

@author: leo
"""

import numpy as np
from math import sqrt
import pandas as pd
from os import listdir
from os.path import isfile, join
import itertools

path = 'C:\\Users\\work\\Documents\\Python_Scripts\\twitter_analytics'
path_ids = 'C:\\Users\\work\\Documents\\Python_Scripts\\twitter_analytics\\ids\\rappers_fr'
path_ids_news_fr = 'C:\\Users\\work\\Documents\\Python_Scripts\\twitter_analytics\\ids\\news_fr'
path_ids_politiques_fr = 'C:\\Users\\work\\Documents\\Python_Scripts\\twitter_analytics\\ids\\politiques_fr'

def create_ids_tab(path_ids_group, group_name):
    '''Creates table of ids in record format for group group_name'''
    print 'Creating ids_tab'
    files = [ f for f in listdir(path_ids_group) if isfile(join(path_ids_group,f)) ]    
    ids_tab = pd.DataFrame(columns = [group_name, 'id'])
    all_ids = dict()
    for file in files:
        f = open(join(path_ids_group, file), 'r')
        ids = f.read().split('\n')
        f.close()
        local_tab = pd.DataFrame()
        local_tab['id'] = ids[:-1]
        local_tab[group_name] = file.replace('_ids.txt', '')
        ids_tab = ids_tab.append(local_tab)
        all_ids[file.replace('_ids.txt', '')] = ids
    print 'Ids_tab created'
    return [all_ids, ids_tab]
   
[all_ids, ids_tab] = create_ids_tab(path_ids_politiques_fr, 'politique')

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

for (key_x, key_y) in itertools.combinations(ids_tab.politique.unique(), 2):

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