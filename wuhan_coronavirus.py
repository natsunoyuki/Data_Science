from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.linear_model import LinearRegression

def linearInversion(x,y):
    N = len(x)
    G = vstack([range(1,N+1,1),ones(N)]).T
    Ginv = dot(linalg.inv(dot(G.T,G)),G.T)
    m = dot(Ginv,y)
    return m

#Population infected with Wuhan virus from 2020 Jan 15. Data from:
#https://graphics.reuters.com/CHINA-HEALTH-MAP/0100B59S39E/index.html   
#Population = array([42,48,65,124,204,295,445,581,845,1315,2019,2800,4579,6058,7815,9820,11948,14551])    

#Population infected with Wuhan virus from 2020 Jan 16. Data from:
#https://en.wikipedia.org/wiki/2019–20_Wuhan_coronavirus_outbreak
Population = array([45,62,121,198,291,440,571,830,1287,1975,2744,4515,5974,7711,9692,11791,14380,17205,20438,24324,28018,31161,34546,37198])
logPopulation = log(Population)
days = arange(1,len(Population)+1,1)

plt.plot(days,Population,'o')
plt.xlabel("Days")
plt.ylabel("Population")
plt.grid("on")
plt.show()

#the power law trend seems to hold true until day 12 (index 11)...
max_lim = 11
mpop = linearInversion(days[:max_lim],logPopulation[:max_lim])
print("Power law factor: {:.3f}".format(exp(mpop)[0]))

#from day 18 (index 17) onwards the increase appears to follow a linear trend...
min_lim = 17
model = LinearRegression()
model.fit(days[min_lim:].reshape(-1,1),Population[min_lim:])
mpop2=[0,0]
mpop2[0] = model.coef_[0]
mpop2[1] = model.intercept_
print("Linear factor: {:.3f}".format(mpop2[0]))

#Make future predictions
x = arange(1,len(Population)+10,1)
ypop = x * mpop[0] + mpop[1]
ypop2 = x * mpop2[0] + mpop2[1]

plt.semilogy(days,Population,"ro")
plt.semilogy(x,exp(ypop),"k")
plt.semilogy(x[min_lim:],ypop2[min_lim:],"b")
plt.xlabel("Days")
plt.ylabel("log_Population")
plt.grid("on")
plt.show()
