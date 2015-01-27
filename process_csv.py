"""shuffle and process the original data"""

import os
import csv
import random

def initiate_categories():
    global category_name, category_value, category_counter
    category_name = header[5:14]
    category_value = [{} for i in range(len(category_name))]
    category_counter = [0 for i in range(len(category_name))]

def update_category_values(row):
    """ we create a dictionnaries to update the category values"""
    for i in range(len(category_name)):
        #update dictionary if needed
        if row[i+5] not in category_value[i]:
            category_value[i][row[i+5]] = category_counter[i]
            category_counter[i] += 1
        #update vector
        row[i+5] = category_value[i][row[i+5]]

def process_header():
    return [header[1]] + header[3:]+ ['day','hour']

def process_row(row):
    update_category_values(row)
    row += [row[2][4:6], row[2][6:8]]
    row = [float(i) for i in row]
    return row

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
                    row = process_row(row)
                    writer.writerow([row[1]]+row[3:])
                _f.close()
            os.remove('.tmp_chunck' + str(i))

shuffle_and_process_big_file('train.csv')
