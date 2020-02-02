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
Population = array([42,48,65,124,204,295,445,581,845,1315,2019,2800,4579,6058,7815,9820,11948,14551])
logPopulation = log(Population)
Deaths = array([2,2,2,3,4,6,9,17,25,41,56,80,106,132,170,213,259,304])
logDeaths = log(Deaths)

days = arange(1,len(Population)+1,1)

mpop = linearInversion(days,logPopulation)
#mdeath = linearInversion(days,logDeaths)

print("Day population power law factor: {:.3f}".format(exp(mpop)[0]))
#print("Day death power law factor: {:.3f}".format(exp(mdeath)[0]))

#Make future predictions
x = arange(1,30+1,1)
ypop = x * mpop[0] + mpop[1]
#ydeaths = x * mdeath[0] + mdeath[1]

plt.plot(days,logPopulation,"ro")
plt.plot(x,ypop,"k")
plt.xlabel("Days")
plt.ylabel("log_Population")
plt.grid("on")
plt.show()