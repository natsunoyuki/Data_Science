from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import curve_fit

def linearInversion(x,y):
    N = len(x)
    G = ones([N,2])
    G[:,0] = x
    Ginv = dot(linalg.inv(dot(G.T,G)),G.T)
    m = dot(Ginv,y)
    return m

#Population infected with Wuhan virus from 2020 Jan 16. Data from:
#https://en.wikipedia.org/wiki/2019–20_Wuhan_coronavirus_outbreak
Population = array([45,62,121,198,291,440,571,830,1287,1975,2744,4515,5974,
                    7711,9692,11791,14380,17205,20438,24324,28018,31161,34546,
                    37198,40171,42638,44653,58761,63851,66492,68500,70548,
                    72436,74185,75003,75891,76288,76936,77150,77658,78064,
                    78497,80026,80151,80270,80409,80552,80651,80695,80735,
                    80754,80778,80793,80813])

logPopulation = log(Population)
days = arange(1,len(Population)+1,1)

#curve fitting to logistic function
def func(x, L, k, x0):
    return L / (1 + exp(-k*(x-x0)))

popt, pcov = curve_fit(func, days, Population)
print("L: {}, k: {:.3f}, x0: {}".format(int(popt[0]),popt[1],int(popt[2])))

#Make future predictions
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
the sigmoid midpoint is x0 = 24 days
and the total population is L = 81072.
China's population is 1.386E9 people. The 5 most affected regions were
Henan with 94E6 people, Hubei with 58.5E6 people, Hunan with 67.37E6 people
Guangdong with 113.46E6 people and Zhejiang with 57.37E6 people.
The total population of these 5 regions is 390700000 people.
This means that this point of time, the total rate of transmission is roughly
0.0002. This should allow us to estimate the value of L.

Case 1: California with population of 39.56E6 people:
L = 0.0002 * 39.56E6 = 8000 people approx.
Case 2: USA with population of 327.2E6 people:
L = 0.0002 * 327.2E6 = 65000 people approx.
Case 3: Japan with population of 126.8E6 people:
L = 0.0002 * 126.8E6 = 25000 people approx.
"""
