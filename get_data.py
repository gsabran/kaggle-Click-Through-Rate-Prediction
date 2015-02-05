import csv
import numpy as np

class _getData:
    """Return the next rows from the test set. Should only be instanciated once."""
    def __init__(self):
        self.datareader = csv.reader(open('train.csv'))
        self.file_ended = False
        next(self.datareader)

    def next(self, n=1):
        """return the next n rows from the training data under the form features, target"""
        data = []
        for i in xrange(n):
            try: l = next(self.datareader)
            except StopIteration:
                print 'Not enough lines: need', n-i, 'more'
                self.file_ended = True
                raise
            data.append([float(i) for i in l])
        data = np.array(data)
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

_get_data = _getData()
def GetData(): return _get_data
