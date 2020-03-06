#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

x0 = -10
x1 = 10
dx = 0.01
x = arange(x0,x1+dx,dx)

#y = random.randn(len(x))
#y = array(sorted(random.randn(len(x))))
#y = y + random.randn(len(y))*0.01*pi

#use logistic map to generate chaotic function
def logmap(x0,A,N):
    x = zeros(N)
    x[0] = x0
    for i in range(N-1):
        x[i+1] = A * x[i] * (1 - x[i])
    return x

y = logmap(0.1,3.7,len(x)*2)
y = y[len(x):]

C = correlate(y, y, mode='full')[len(y)-1:]
C = C / max(abs(C))

plt.figure(figsize=[10,5])
plt.plot(x,y)
plt.show()

plt.figure(figsize=[10,5])
plt.plot(x,C)
plt.show()

#Auto-regression model:
M = len(y)
N = 10 #number of features to create (currently one week before...)
i = N
R = zeros([M-N,N])
Y = y[N:]
while i < M:
    R[i-N] = y[i-N:i]
    i=i+1

#linear regression:
days = arange(1,len(Y)+1)
length1 = int(len(Y)*0.75)
R_train = R[:length1]
R_test = R[length1:]
Y_train = Y[:length1]
Y_test = Y[length1:]
days_train = days[:length1]
days_test = days[length1:]
print("R: {}, R_train: {}, R_test: {}".format(shape(R),shape(R_train),shape(R_test)))
print("Y: {}, Y_train: {}, Y_test: {}".format(shape(Y),shape(Y_train),shape(Y_test)))
print("days: {}, days_train: {}, days_test: {}".format(shape(days),shape(days_train),shape(days_test)))

model = LinearRegression()
model.fit(R_train,Y_train)

#m = hstack([model.coef_,model.intercept_])
print("Coefficients: {};\nIntercept: {}".format(model.coef_,model.intercept_))
print("Train Accuracy: {:.3f}".format(model.score(R_train,Y_train)))
print("Test Accuracy: {:.3f}".format(model.score(R_test,Y_test)))
print("data +1 day prediction: {}".format(model.predict(Y[-N:].reshape(1,-1))))

plt.figure(figsize=[10,5])
plt.plot(days,Y)
plt.plot(days_train,Y_train)
plt.plot(days_test,Y_test)
plt.plot(days_train,model.predict(R_train),"-.")
plt.plot(days_test,model.predict(R_test),"-.")
plt.legend(["Data","Train_data","Test_data","Train_pred","Test_pred"])
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()

