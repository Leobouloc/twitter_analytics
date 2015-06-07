# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 09:56:30 2015

@author: leo
"""

import pandas as pd
from os.path import join
from time import sleep
import twitter

from access_keys import key, secret, key_access, secret_key


def init_new_table(path, list_text_name):
    print 'creating csv table...'
    f = open (join(path, list_text_name))
    text = f.read()
    f.close()
    api = twitter.Api(consumer_key = key,
                              consumer_secret = secret,
                              access_token_key = key_access,
                              access_token_secret = secret_access)   
    rappers_list = text.split('\n')
    rapper_dict_list = []
    for x in rappers_list:
        print x
        rapper_dict = dict()
        rapper_dict['name'] = x.split(' : ')[0]
        rapper_dict['twitter'] = x.split(' : ')[1]
        a = api.GetUser(screen_name = rapper_dict['twitter'])
        rapper_dict['followers_count'] = a.AsDict()['followers_count']
        rapper_dict_list += [rapper_dict]
#    import pdb
#    pdb.set_trace()
    table = pd.DataFrame()
    for x in rapper_dict_list:
        print x
        table.loc[x['name'], 'twitter'] = x['twitter']
        table.loc[x['name'], 'followers_count'] = x['followers_count']

    table.to_csv(join(path, list_text_name.replace('.txt', '.csv')), sep = ';')
    print 'csv table created'

######################################################

def read_cursor(path_cursors, twitter_name):
    try: 
        f = open(join(path_cursors, twitter_name + '_cursor.txt'), 'r')
        cursor = f.read().split('\n')[-1]
        f.close()
    except:
        f = open(join(path_cursors, twitter_name + '_cursor.txt'), 'w')
        f.write('-1')
        cursor = '-1'
        f.close()
    return str(cursor)

def append_cursor(path_cursors, twitter_name, cursor):
    f = open(join(path_cursors, twitter_name + '_cursor.txt'), 'a')
    f.write('\n' + str(cursor))
    f.close()

def append_ids(path_ids, twitter_name, ids):
    try: 
        f = open(join(path_ids, twitter_name + '_ids.txt'), 'a')
        for x in ids:
            f.write(str(x) + '\n')
        f.close()
    except:
        f = open(join(path_ids, twitter_name + '_ids.txt'), 'w')
        for x in ids:
            f.write(str(x) + '\n')
        f.close()
    

def step(path_cursors, path_ids, twitter_name):
    api = twitter.Api(consumer_key = key,
                          consumer_secret = secret,
                          access_token_key = key_access,
                          access_token_secret = secret_access)
                          
    cursor = read_cursor(path_cursors, twitter_name)                    
    a = api.GetFollowerIDsPaged(screen_name = twitter_name, cursor = cursor)
    ids = a[2]['ids']
    cursor = a[2]['next_cursor']
    append_ids(path_ids, twitter_name, ids)
    append_cursor(path_cursors, twitter_name, cursor)
    print cursor
    return str(cursor) != '0'


if __name__ == '__main__':
    
    to_load = 'politiques_fr'
    path = 'C:\\Users\\work\\Documents\\Python_Scripts\\twitter_analytics'
    path_cursors = join(path, 'cursors', to_load)
    path_ids = join(path, 'ids', to_load)
    
    
    twitter_name = 'eminem'

    while True:
        try:
            try:
                table = pd.read_csv(join(path, to_load + '.csv'), sep = ';')  
            except:
                init_new_table(path, to_load + '.txt')
                table = pd.read_csv(join(path, to_load + '.csv'), sep = ';') 
            for twitter_name in table.twitter:
                print '  >>>', twitter_name
                try:
                    f = open(join(path_cursors, twitter_name + '_cursor.txt'), 'r')
                    text = f.read()
                    f.close()
                    if '0' not in text.split('\n'):
                        go = True
                    else:
                        go = False
                except:
                    go = True
                    
                while go:
                    go = step(path_cursors, path_ids, twitter_name)
                    sleep(60)
        except:
            sleep(60)
            
    



        
    