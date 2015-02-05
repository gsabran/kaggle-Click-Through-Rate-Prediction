#import os
#os.chdir('C:/Users/Pc-stock2/Desktop/Kaggle/kaggle-Click-Through-Rate-Prediction')

#from sklearn.linear_model import LogisticRegression
from predict import *
from get_data import *
from nn import *

#lr = LogisticRegression()
#lr.fit(X_train, Y_train)
#predict_on_test_data(lr)

data_getter = GetData()
nn=neural_network()
iteration=0
while (data_getter.file_ended is False):
    if (iteration%100 ==0): print 'row number' + str(iteration*5000)
    X_train, Y_train = data_getter.next(n=5000)
    if (nn.initialized is False): nn.initialize_parameters(len(X_train[0]))
    nn.fit(X_train,Y_train)
    iteration +=1

nn.print_cost_checking()
nn.theta
predict_on_test_data(nn)