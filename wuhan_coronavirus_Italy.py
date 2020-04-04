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

#Italian population infected with Wuhan virus from 2020 Jan 31. Data from:
#https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Italy
Population = array([2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                    20,79,150,227,320,445,650,888,1128,1694,2036,
                    2502,3089,3858,4636,5883,7375,9172,10149,12462,
                    15113,17660,21157,24747,27980,31506,35713,41035,
                    47021,53578,59138,63927,69176,74386,80539,86498,
                    92472,97689,101739,105792,110574,115242,119827])
    
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
