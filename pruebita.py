# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

# #Funcion a fittear
# def func(x, a, b, c):
#     return a + b*x + c*x*x


# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise




# print curve_fit(func, xdata, ydata, p0=None,sigma = None)
# popt, pcov = curve_fit(func, xdata, ydata,p0=None,sigma = None)
# plt.plot(xdata, ydata, 'b-', label='data')
# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Funcion a fittear
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fitFunc(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[0]*x[1]

print np.array([[1,2,3,4,6],[4,5,6,7,8]])

x_3d = np.array([[1,2,3,4,6],[4,5,6,7,8]])
y= np.array([4,8,3,6,1])

popt, pcov = curve_fit(fitFunc, x_3d, y, p0=None)

plt.plot(np.array(list(range(5))),y, 'b-', label='data')

#plt.plot(np.array(list(range(5))), fitFunc(x_3d, *popt), 'r-', label='fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

