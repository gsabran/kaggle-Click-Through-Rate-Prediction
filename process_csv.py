# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:32:47 2015

@author: Pc-stock2
"""

import os
os.chdir('C:/Users/Pc-stock2/Desktop/Kaggle')


def update_category_values(row_number,x):
    """ we create a dictionnaries to update the category values"""
    if (row_number==0):
        global category_name, category_value, category_counter
        category_name=names[5:14]
        category_value=[{} for i in range(len(category_name))]
        category_counter=[0 for i in range(len(category_name))]
    
    for i in range(len(category_name)):
        #update dictionary if needed
        if x[i+5] not in category_value[i]:
            category_value[i][x[i+5]]=category_counter[i]
            category_counter[i]+=1
        #update vector
        x[i+5]=category_value[i][x[i+5]]




def process_csv(raw_data, seed, p, train_set, valid_set):
    """ create a training set and validation set from the raw data """
    import csv
    import random

    random.seed(seed)    
    
    i = open(raw_data)
    o1 = open(train_set,'wb')
    o2 = open(valid_set,'wb')
    
    reader = csv.reader(i)
    writer1 = csv.writer(o1)
    writer2 = csv.writer(o2)
    
    global names #save the feature names
    names=next(reader)
    
    for row_number, row in enumerate(reader):
        #update category values
        update_category_values(row_number,row)
        #divide YYMMDDHH into two seperate features 'day' and 'hour' 
        #since 'year' and 'month' never change
        row += [row[2][4:6],row[2][6:8]]
        #make sure all values are floats - will be useful later
        row = [float(i) for i in row]
        
        #write the headers for the output files
        #we don't write 'click' - the output - in a seperate file as it takes too much space
        #we remove the raw 'id' and 'hour' as we don't use them
        if (row_number==0): 
            writer1.writerow([names[1]] + names[3:]+ ['day','hour'])
            writer2.writerow([names[1]] + names[3:]+ ['day','hour'])

        #seperate the two sets according to p
        r=random.random()
        if (r<p):
            writer1.writerow([row[1]]+row[3:])
        else:
            writer2.writerow([row[1]]+row[3:])
            
        #keep track every 500,000            
        if (row_number%500000==0): print 'ok'
    
    o1.close()
    o2.close()

####################################################
####################################################
#
process_csv('raw_data.csv', 1, 0.9, 'train.csv', 'valid.csv')
