import numpy as np 
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
from dateutil.parser import parse
from random import shuffle
from sklearn.linear_model import LinearRegression



def add_intersection(X):
	l = len(X)
	X = np.concatenate((np.ones((l,1)),X),axis = 1)
	return X	


def linear_regression(X,Y,gamma=0):#gamma ridge regularization parameter
	l = len(X)
	X = add_intersection(X)
	temp = np.dot(np.transpose(X),X)
	temp = temp+gamma*np.identity(X.shape[1])
	temp1 = np.dot(np.transpose(X),Y)
	beta = np.dot(inv(temp),temp1)
	return beta

def linear_predict(X,beta):
	X = add_intersection(X)
	return np.dot(X,beta)

def RSS(X,Y,beta,gamma=0):
	predict = linear_predict(X,beta)
	sum1 = np.sum((Y-predict)**2)
	sum2 = gamma*np.sum(beta[1:]**2)
	return sum1+sum2 
def Rsquare(X,Y,beta):
	temp1 = RSS(X,Y,beta)
	temp2 = np.sum((Y-Y.mean())**2)
	return 1.0-1.0*temp1/temp2





def five_fold(X,Y,gamma=0):	
	l = len(X)
	subset_size = l/5
	index = list(range(l))
	shuffle(index)
	X = X[index]
	Y = Y[index]
	RR = np.zeros(5)
	Rss = np.zeros(5)
	for i in range(5):
		testX = X[i*subset_size:(i+1)*subset_size,:]
		testY = Y[i*subset_size:(i+1)*subset_size,:]
		trainX = np.vstack((X[:i*subset_size,:],X[(i+1)*subset_size:,:]))
		trainY = np.vstack((Y[:i*subset_size,:],Y[(i+1)*subset_size:,:]))
		beta = linear_regression(trainX,trainY,gamma)
		RR[i] = Rsquare(trainX,trainY,beta)
		Rss[i] = RSS(testX,testY,beta)/len(testX)
	return RR.mean(),np.sqrt(Rss.mean())

def get_data():
	file = pd.read_csv("http://files.zillowstatic.com/research/public/Neighborhood/Neighborhood_Zhvi_SingleFamilyResidence.csv")
	data = file[ file.City == 'Providence']
	data1 = data[data.RegionName == 'Blackstone']
	data2 = data[data.RegionName == 'College Hill']
	date = list(file.columns.values)[7:]
	X = data1.values[0,7:]
	Y = data2.values[0,7:]
	X = X.reshape((len(X),1))
	Y = Y.reshape((len(Y),1))
	X = X.astype(float)
	Y = Y.astype(float)
	return X,Y,date

X, Y, date = get_data()	
date = pd.to_datetime(date)
plt.figure()
plt.plot(date,X,label = 'Blackstone')
plt.plot(date,Y,label = 'College Hill')
plt.legend()
plt.show()



clf = LinearRegression()
clf.fit(X,Y)
print 'rsquare from package', clf.score(X,Y)
rsquare, average_error = five_fold(X,Y)
print 'rsquare estimator is', rsquare
print 'average error prediction is', average_error

