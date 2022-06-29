import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def ddm(n,s,sample_size):
    "n = number of dice, s = sides of dice, sample_size = sample_size"
    sample_size = int(sample_size)
    min_value = 1
    max_value = s +1
    size = (n,sample_size)
    sample_bins = np.arange(n,s*n+3,step = 1)
    
    sample = np.random.randint(min_value,max_value,size)
    sample_sum = np.sum(sample,axis = 0)
    sample_hist = np.histogram(sample_sum,bins = sample_bins,density = True)[0]
    
    

    return(sample_bins[1:],sample_hist)

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

def dice_data(n,m,mod = 0,sample = 1e6,plot = False):
    
    data = ddm(n,m,10000)
    try:
        
        H,A,x0,sigma = gauss_fit(data[0],data[1])
        
    except:
        print("No Gauss fit found")
        H,A,x0,sigma = 1,1,1,1
    if plot == True:
        x = np.linspace(data[0][0],data[0][-1],100)
        y = gauss(x,H,A,x0,sigma)
        plt.plot(data[0],data[1])
        plt.plot(x,y)
        x0 += mod
        print(f'\nH: {H}\nA: {A}\nx0: {x0}\nsigma: {sigma}\n')
        plt.show()
    else:
        x0 += mod
    return(H,A,x0,sigma)


def create_data():
    final_data = []
    N = np.arange(3,100)
    M = np.arange(3,100)
    L = (100-3)**2
    i = 0
    for n in N:
        for m in M:
            print(f"\r{100*i/L:.3f}%",end = "\r")
            H,A,x0,sigma = dice_data(n,m)
            final_data.append([n,m,H,A,x0,sigma])
    final_data = np.array(final_data)
    np.save("Dice.npy",final_data)
    

def theory(n,s):
    sigma = np.sqrt(n*(s**2-1)/12)
    return(sigma)

diff = []
for n in np.arange(3,10):
    for s in np.arange(4,12):
        
        s1 = dice_data(4,8,mod = 0,sample = 1e6,plot = False)[-1]
        s2 = theory(4,8)
        diff.append(s2-s1)
plt.plot(diff)
plt.show()
