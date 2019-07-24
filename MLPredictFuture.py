#this script shows how using expert knowledge about a time series can lead
#to proper predictions of the future
from scipy import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

X=arange(0,10,0.01)
y=cos(2*pi*X)+random.randn(len(X))*0.2
X=X.reshape(-1,1)
#X_cyc contains the range of x where y is periodic
X_cyc=X[0:100]
for i in range(9):
    X_cyc=vstack([X_cyc,X[0:100]])

reg=RandomForestRegressor(n_estimators=100,random_state=0)

#split the original feature into training and test sets 7:3
n_train=700
X_train,X_test=X[:n_train],X[n_train:]
#split the cyclic features
X_cyc_train,X_cyc_test=X_cyc[:n_train],X_cyc[n_train:]
#split the target into training and test sets
y_train,y_test=y[:n_train],y[n_train:]

#fit the random forest to the training data set
#reg.fit(X_cyc_train,y_train)
reg.fit(X_train,y_train)

#if we use the original X features in the prediction, we will get a simple
#constant line as the Machine cannot make predictions of the future using
#linearly increasing feature sets.
y_pred=reg.predict(X_test)
#on the other hand when we observe that the cosine function is cyclic in pi
#we can use this knowledge to "create" a new cyclic feature set which will
#allow the Machine to predict the future!
y_pred_train=reg.predict(X_cyc_train)
y_pred_test=reg.predict(X_cyc_test)

#what the Machine is most likely doing is discovering a "pattern" between the
#cyclic pi feature and the cyclic cosine function........

plt.plot(X_train,y_train,'b')
plt.plot(X_test,y_test,'r')
plt.plot(X_train,y_pred_train,'y-.')
plt.plot(X_test,y_pred_test,'g-.')
plt.plot(X_test,y_pred,'k-.')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(["y_train","y_test","y_pred_train","y_pred_test","y_pred"])
plt.show()
