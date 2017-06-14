import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Funcion a fittear
def func(x, a, b, c):
    return a + b*x + c*x*x


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise




print curve_fit(func, xdata, ydata, p0=None,sigma = None)
popt, pcov = curve_fit(func, xdata, ydata,p0=None,sigma = None)
plt.plot(xdata, ydata, 'b-', label='data')
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


