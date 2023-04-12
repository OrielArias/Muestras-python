import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import misc

V=np.array([0.3,0.3,0.71,0.71,0.73,0.76,0.77,0.78,0.79])
I=np.array([0,1,2.5,3,4,8,12,16,20])
plt.plot(V,I)
plt.xlabel('$V$ (V)')
plt.ylabel('$I$ (mA)')
plt.title('Curva I-V de diodo')

q=1.6e-19
k=1.38e-23

def diodo(V,I_0,T):
  I=I_0*(np.exp((q*V)/(k*T))-1)
  return I
fit=optimize.curve_fit(diodo,V,I,[0,350])

fit_I_0=fit[0][0]
fit_T=fit[0][1]

plt.plot(V,I,label='Datos medidos')
plt.plot(V,diodo(V,fit_I_0,fit_T),label='Fit')
plt.xlabel('$V$ (V)')
plt.ylabel('$I$ (mA)')
plt.legend()
plt.title('Curva I-V de diodo con fiteo')

print(fit_I_0)
print(fit_T)
 
def mse (actual, pred):
    actual, pred = np.array (actual), np.array (pred)
    return np.square (np.subtract (actual, pred)). mean () 
mse(I,diodo(V,fit_I_0,fit_T))

R=1000
def f(i):
  return V_in-np.log((i/fit_I_0) + 1)*k*fit_T/q - i*R

V_list=np.linspace(0.275,1.5,50)

I_newton=np.zeros_like(V_list)
x0=0
for j in range(len(V_list)):
  V_in=V_list[j]
  I_newton[j]=optimize.newton(f,x0)

V_out=I_newton*R

V_in_med=np.array([0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5])
V_out_med=np.array([0,0,0,0.035,0.096,0.171,0.253,0.34,0.43,0.522,0.615,0.71,0.803,0.9])

plt.plot(V_in_med,V_out_med,label='Datos medidos')
plt.plot(V_list,V_out,label='Teorico')
plt.legend()
plt.xlabel('$V_{in}$ [V]')
plt.ylabel('$V_{out}$ [V]')
plt.title('$V_{out}$ v/s $V_{in}$')

dV_out=np.gradient(V_out_med)
d2V_out=np.gradient(dV_out)
plt.scatter(V_in_med,d2V_out)
plt.xlabel('$V_{in}$ (V)')
plt.ylabel('$\\frac{d^2V_{out}}{dV_{in}}$')
plt.title('Curvatura de $V_{out}$')

d3V_out=np.gradient(d2V_out)
d4V_out=np.gradient(d3V_out)

x=np.linspace(0,1.5,100)
V0=0.9
taylor=0.34+dV_out[8]*(x-V0)+(d2V_out[8]*(x-V0)**2)/np.math.factorial(2)+(d3V_out[8]*(x-V0)**3)/np.math.factorial(3)+(d4V_out[8]*(x-V0)**4)/np.math.factorial(4)
plt.plot(x,taylor)

plt.xlabel('$V_{in}$')

plt.ylabel('$V_{out}$')
plt.title('Taylor de orden 4 en torno $V=0.9$ V')

offset=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]

potencia=[-74,-72,-70,-55,-38.8,-30.8,-27.6,-26.4,-25.60,-25.2,-25.2,-25.2,-25.2,-25.2,-25.2,-25.2]

plt.plot(offset,potencia)
plt.xlabel('DC offset (V)')
plt.ylabel('Potencia (dB)')
plt.title('Potencia de 2do armonico vs DC offset')