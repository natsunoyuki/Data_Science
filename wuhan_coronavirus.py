from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg

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
Population = array([45,62,121,198,291,440,571,830,1287,1975,2744,4515,5974,7711,9692,11791,14380,17205,19655])
logPopulation = log(Population)


days = arange(1,len(Population)+1,1)

mpop = linearInversion(days,logPopulation)

print("Day population power law factor: {:.3f}".format(exp(mpop)[0]))

#Make future predictions
x = arange(1,len(Population)+1,1)
ypop = x * mpop[0] + mpop[1]

plt.semilogy(days,Population,"ro")
plt.semilogy(x,exp(ypop),"k")
plt.xlabel("Days")
plt.ylabel("log_Population")
plt.grid("on")
plt.show()
