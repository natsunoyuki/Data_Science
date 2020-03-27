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

#Japan Population infected with Wuhan virus from 2020 Jan 24. Data from:
#https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Japan
Population = array([3,3,4,6,7,11,14,17,20,20,20,20,23,25,25,25,25,26,26,28,33,
                    41,53,59,65,73,85,93,105,132,144,157,164,186,210,230,239,
                    254,268,284,317,349,408,455,488,514,568,620,675,716,780,
                    814,829,873,914,950,1007,1054,1086,1128,1193,1291,1364])

logPopulation = log(Population)
days = arange(1,len(Population)+1,1)

#the power law trend seems to hold true until day 12 (index 11)...
#max_lim = 11
#mpop = linearInversion(days[:max_lim],logPopulation[:max_lim])
#print("Power law coefficients: {}".format(exp(mpop)))

#from day 18 - 24 (index 17) onwards the increase appears to follow a linear trend...
#min_lim2 = 17
#max_lim2 = 24
#mpop2 = linearInversion(days[min_lim2:max_lim2],Population[min_lim2:max_lim2])
#model = LinearRegression()
#model.fit(days[min_lim:].reshape(-1,1),Population[min_lim:])
#mpop2=[0,0]
#mpop2[0] = model.coef_[0]
#mpop2[1] = model.intercept_
#print("Linear coefficients: {}".format(mpop2))

#curve fitting to logistic function
def func(x, L, k, x0):
    return L / (1 + exp(-k*(x-x0)))

popt, pcov = curve_fit(func, days, Population)
print("Logistic coefficients: {}".format(popt))

#Make future predictions
x = arange(1,len(Population)+14,1)
#ypop = x * mpop[0] + mpop[1]
#ypop2 = x * mpop2[0] + mpop2[1]
logistic_prediction = func(x,*popt)

#plt.semilogy(days,Population,"ro")
#plt.semilogy(x[:max_lim+10],exp(ypop[:max_lim+10]),"k")
#plt.semilogy(x[min_lim2:max_lim2+10],ypop2[min_lim2:max_lim2+10],"b")
#plt.semilogy(x,logistic_prediction,"c")
#plt.xlabel("Days")
#plt.ylabel("log_Population")
#plt.grid("on")
#plt.legend(["Observations","Power Law","Linear","Logistic"])
#plt.show()

plt.plot(days,Population,"ro")
plt.plot(x,logistic_prediction,"k")
plt.xlabel("Days")
plt.ylabel("Population")
plt.grid("on")
plt.legend(["Observations","Logistic Function"])
plt.show()