from sklearn.linear_model import LogisticRegression
from predict import *
from get_data import *

data_getter = GetData()
X_train, Y_train = data_getter.next(n=10000)
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predict_on_test_data(lr)
