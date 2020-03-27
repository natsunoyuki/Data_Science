from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import curve_fit

#linear inversion
def linearInversion(x,y):
    N = len(x)
    G = ones([N,2])
    G[:,0] = x
    Ginv = dot(linalg.inv(dot(G.T,G)),G.T)
    m = dot(Ginv,y)
    return m

#curve fitting to logistic function
def func(x, L, k, x0):
    return L / (1 + exp(-k*(x-x0)))

#USA population infected with Wuhan virus from 2020 Jan 21. Data from:
#https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_the_United_States
#Population = array([1,1,1,2,3,5,5,5,5,6,7,8,11,11,11,12,12,12,12,12,12,12,12,
#                    12,12,12,12,12,12,12,14,14,14,14,14,14,15,15,19,24,42,57,
#                    85,111,175,252,353,497,645,936,1205,1598,2163,2825,3496,
#                    4372,5656,8074,12018,17438,23573])

Population = array([12,12,12,12,12,12,12,14,14,14,14,14,14,15,15,19,24,42,57,
                    85,111,175,252,353,497,645,936,1205,1598,2163,2825,3496,
                    4372,5656,8074,12018,17439,23710,32341,42749,52685,64916,
                    85435])

    
logPopulation = log(Population)
days = arange(1,len(Population)+1,1)

popt, pcov = curve_fit(func, days, Population)
print("L: {}, k: {:.3f}, x0: {}".format(int(popt[0]),popt[1],int(popt[2])))

#Make future predictions 1 month ahead
x = arange(1,len(Population)+30,1)
logistic_prediction = func(x,*popt)

plt.plot(days,Population,"ro")
plt.plot(x,logistic_prediction,"k")
plt.xlabel("Days")
plt.ylabel("Population")
plt.grid("on")
plt.legend(["Observations","Logistic Function"])
plt.show()

"""
Post-analysis comments: 
It would seem that the logistic growth rate is k = 0.224 per day,
the sigmoid midpoint is x0 = 47 days
and the total population is L = 70360.
Surprisingly, the logistic growth rate of k = 0.224 per day is very 
close to that of the Chinese data....
"""
