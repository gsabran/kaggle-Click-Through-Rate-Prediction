"""
shuffle and process the original data. This takes a while to run but should just be done once.
Need: train data under 'train.csv' and test data under 'test.csv' Files will be replaced.
"""

import os
import csv
import random

def initiate_categories():
    global category_name, category_value, category_counter
    category_name = header[5:14]
    category_value = [{} for i in range(len(category_name))]
    category_counter = [0 for i in range(len(category_name))]

def update_category_values(row, include_target):
    """ we create a dictionnaries to update the category values"""
    for i in range(len(category_name)):
        #update dictionary if needed
        idx = include_target+i+4
        if row[idx] not in category_value[i]:
            category_value[i][row[idx]] = category_counter[i]
            category_counter[i] += 1
        #update vector
        row[idx] = category_value[i][row[idx]]

def process_header():
    return [header[1]] + header[3:]+ ['day','hour']

def process_row(row, include_target=True):
    update_category_values(row, include_target)
    row += [row[include_target+1][4:6], row[include_target+1][6:8]]
    if include_target: row = [float(i) for i in row]
    else: row = [row[0]] + [float(i) for i in row[1:]]
    return [row[include_target]]+row[include_target+2:]

def shuffle_and_process_big_file(filename, seed=123, n_chunck=1000):
    """ shuffle the file """
    # separate the big file in small files randomly
    _tmp = [open('.tmp_chunck' + str(i), 'wb') for i in range(n_chunck)]
    _writers = [csv.writer(f) for f in _tmp]
    with open(filename) as source:
        reader = csv.reader(source)
        global header
        header = next(reader)
        for l in reader: _writers[int(random.random() * n_chunck)].writerow(l)
    os.remove(filename)
    for f in _tmp: f.close()
    # shuffle each of the small files and merge them
    with open(filename, 'wb') as target:
        writer = csv.writer(target)
        writer.writerow(process_header())
        initiate_categories()
        for i in range(n_chunck):
            with open('.tmp_chunck' + str(i)) as _f:
                _reader = csv.reader(_f)
                data = [(random.random(), row) for row in _reader]
                data.sort()
                for _, row in data:
                    writer.writerow(process_row(row))
                _f.close()
            os.remove('.tmp_chunck' + str(i))
    # convert the test file in a similar way
    with open('test.csv') as f, open('.test.csv', 'wb') as f2:
        reader = csv.reader(f)
        writer = csv.writer(f2)
        writer.writerow(['id'] + process_header()[1:])
        next(reader)
        for row in reader:
            writer.writerow(process_row(row, include_target=False))
    os.rename('.test.csv', 'test.csv')

shuffle_and_process_big_file('train.csv')
