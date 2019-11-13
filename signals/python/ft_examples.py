import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal

def signal_representation(sig,N,t_obs, filename, func, cut):
    t = np.linspace(-t_obs, t_obs, N)
    #plt.subplot(2,1,1)
    # plot time signal:
    plt.plot(t,sig.real)
    plt.title("temporal signal of " + func)
    plt.ylabel('Values')
    plt.xlabel("$t$")
    plt.savefig(filename+".png")
    plt.show()
    plt.close()
    
    #plt.subplot(2,1,2)
    #plot frequency signal
    f=(np.arange(N)-N/2)
    f_sub = f[(cut / 2 - 1)*N/cut:(cut / 2 + 1 )*N/cut]
    fft_sig = (np.fft.fftshift(np.fft.fft(sig,N)/N))
    fft_sub = fft_sig[(cut / 2 - 1)*N/cut:(cut / 2 + 1 )*N/cut]
    plt.plot(f_sub, fft_sub) 
    #plt.plot(f,np.abs(fft_sig))
    plt.title("frequenciel signal of " + func)
    plt.ylabel("Values")
    plt.xlabel("$\lambda$")
    plt.savefig('fft'+filename+'.png')
    plt.show()

def square(x) : 
    sig = np.zeros(len(x))
    for i in range(len(x)) : 
        if x[i] > -np.pi and x[i] < np.pi:
            sig[i] = 1
        else:
            sig[i] = 0
    return sig


N = 512 
t = np.linspace(-np.pi,np.pi,N) 
sig = np.cos(3*t) 
signal_representation(sig, N, np.pi, 'cos3t', '$\cos(3t)$', 32)


t = np.linspace(-2*np.pi, 2*np.pi, N)
sig_2  = square(t)
signal_representation(sig_2, N, 2*np.pi, 'Rectangular-Wave', 'Rectangle function', 32)

t = np.linspace(-1, 1, N)
sig_2  = np.cos(6*np.pi*t)*np.exp(-np.pi*t**2)
signal_representation(sig_2, N, 1, 'cos-exp', '$\cos(6\pi t)e^{-\pi t^2}$', 32)















