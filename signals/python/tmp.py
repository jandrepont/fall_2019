import matplotlib.pyplot as plt
import numpy as np 

N = 2048
lam = np.linspace(-10, 10, N)
sig = np.sin(2*lam*np.pi**2)  
sig = sig / ((np.pi)*lam) 
plt.plot(lam, sig) 
plt.title("Explicit FT of Rectangle function")
plt.ylabel('Values')
plt.xlabel("$t$")
plt.savefig('ft-rectangle.png') 
plt.show()
#sig = np.sqrt(2)*np.sin(lam*np.pi) 
#sig = sig / (np.sqrt(np.pi)*lam)
#plt.plot(lam, sig) 
#plt.show()
