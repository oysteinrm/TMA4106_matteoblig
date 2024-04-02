import numpy as np
import matplotlib.pyplot as plt

#Løsning med laplas og transferfunksjon
timeVec = np.linspace(0, 48, 10000) #tidsvektor
omega = 2*np.pi / 24 #vinkelfrekvens som gir en periode på 24 timer
alpha = 1 

def kompleksFourierFirkantpuls(N, timeVec, high, low, omega): 
    f = 0 
    amplitude = high - low

    for n in range(-N, N):
        f += amplitude/(2*n-1) * np.exp(1j * (2*n-1) * timeVec * omega)
    f *= 1 / (1j * np.pi)
    return f + (high + low) / 2 #skyver grafen oppover

def losning(N, timeVec, high, low, omega, alpha): 
    f = 0 
    amplitude = high - low
    #ganger firkantpusen med H(i*omega)
    for n in range(-N, N):
        f += amplitude/(2*n-1) * np.exp((2*n-1) * 1j * timeVec * omega) * 1 /(1j * (2*n-1) * omega + alpha)
    f *= 1 / (np.pi * 1j)
    return f + (high + low) / 2 * alpha

firkant = kompleksFourierFirkantpuls(1000, timeVec, 22, 18, omega)
y = losning(1000, timeVec, 22, 18, omega, alpha)

plt.plot(timeVec, firkant, label='Temperatur Bad')
plt.plot(timeVec, y, label='Temperatur Ølfat')

plt.title('Løsning med laplas og transerfunksjon')
plt.xlabel('Tid [timer]')
plt.ylabel('Temperatur [grader celsius]')
plt.legend()
plt.grid(True)
plt.show()

#Løsning med partikulær og homogen løsnging
T = 24
omega = 2 * np.pi / T

c_0 = 2
def c(n):
    return 4 / (1j * (2*n-1) * np.pi)

#firkantpuls fra 18-22
def f(t):

    f_sum=c_0
    for n in range(-1000, 1000):
            f_sum += c(n) * np.exp(1j * (2*n-1) *omega * t)
    
    return f_sum+18

alpha=1
y_0=18

def A(n):
    return (4*np.exp(1j*(2*n-1)*omega/2*t))/(1j*np.pi*alpha*(2*n-1)-np.pi*(2*n-1)**2*omega/2)

def y(t):
    y_sum=0
    partikulær=0
    for n in range(-1000, 1000):
        y_sum-=A(n)
        partikulær+=A(n)*np.exp(1j*(2*n-1)*omega/2*t)
    y_sum+=y_0-20/alpha
    y_sum*=np.exp(-alpha*t)
    y_sum+=20/alpha+partikulær
    return y_sum

t = np.linspace(0, 48, 10000)

plt.figure()
plt.plot(t, np.real(f(t)),label="Temperatur Bad")
plt.plot(t,y(t),label="Temperatur Ølfat")
plt.title('Løsning med partikulær og homogen løsnging')
plt.xlabel('Tid [timer]')
plt.ylabel('Temperatur [grader celsius]')
plt.legend()
plt.grid(True)
plt.show()

