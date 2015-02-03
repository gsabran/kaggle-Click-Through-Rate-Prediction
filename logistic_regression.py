#from sklearn.linear_model import LogisticRegression
from predict import *
from get_data import *
from nn import *

data_getter = GetData()
X_train, Y_train = data_getter.next(n=10000)
#lr = LogisticRegression()
#lr.fit(X_train, Y_train)
#predict_on_test_data(lr)
nn=neural_network()
nn.initialize_parameters(len(X_train[0]))
nn.fit(X_train,Y_train,1)
predict_on_test_data(nn)
