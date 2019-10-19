import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def bandai(x):
    return x>0

x = np.linspace(-8,8,1001)
plt.gca(yticks=np.linspace(0,1,11),xlabel='x',ylabel='h')
plt.plot(x,bandai(x),'r',lw=3)
plt.plot(x,sigmoid(x),'g',lw=3)
plt.grid(ls=':')
plt.show()