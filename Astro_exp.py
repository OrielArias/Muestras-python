import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
###Parte 1: Hot-Cold Test
##Definición de datos
T_hot = 300
T_cold = 77
P_hot_dB = -44.5
P_cold_dB = -47.94
##Conversión de dBm a Watts
P_hot = 10**((P_hot_dB-30)/10)
P_cold = 10**((P_cold_dB-30)/10)
print(f'La potencia Hot es {P_hot} [W].')
print(f'La potencia Cold es {P_cold} [W].')
##Cálculo de Y-factor
Y_factor = P_hot/P_cold
print(f'El Y_Factor es {Y_factor} .')
##Cálculo de temperatura de ruido de antena (T_rec)
T_rec = (T_hot-T_cold*Y_factor)/(Y_factor-1)
print('La temperatura de ruido del receptor es de ' + str(T_rec) + ' K')
###Parte 2: Antenna Dipping
##Definición de datos
Hot_ref_dB = -44.54
Z = np.array([66.5, 73.4, 77.16, 79.52, 81.15, 82.34, 83.24, 83.96, 84.53, 85.01])
P_sky_dB = np.array([-45.56, -45.22, -45.01, -44.88, -44.8, -44.73, -44.71, -44.67, -44.64, -44.63])
##Obtengo vector de secante para no escribir decimal por decimal. (parte de definición de datos)
Sec = np.zeros(10)
for x in range(0,len(Z)):
  Sec[x]= 1/(m.cos(m.radians(Z[x])))
#print(Sec)
##Conversión de dBm a Watts
Hot_ref = 10**((Hot_ref_dB-30)/10)
#print(Hot_ref)
P_sky = np.zeros(10)
y_ref = np.zeros(10)
for x in range(0,len(P_sky_dB)):
  P_sky[x] = 10**((P_sky_dB[x]-30)/10)
#print(P_sky)
##Generación de vectores para plotear
y = np.zeros(10)
for x in range(0,len(P_sky)):
   y[x] = m.log(abs(P_sky[x]-Hot_ref))
#print(y)
##Ploteo de datos y regresión lineal
pend, coef = np.polyfit(Sec,y,1)
x_poly = np.linspace(Sec[0],Sec[-1],len(Sec))
plt.figure()
plt.plot(Sec,y)
plt.plot(x_poly,pend*x_poly+coef)
plt.xlabel('1/cos(Z)')
plt.ylabel('ln(\u0394W)')
plt.title('Gráfico de opacidad zenital \u03C40')
plt.legend(['Datos obtenidos','Regresión lineal'])
plt.show()
print('La opacidad zenital es de ' + str(-pend))