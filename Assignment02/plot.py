#!/usr/bin/python3
import numpy as np 
import matplotlib.pyplot as plt
from math import *

x = np.array(range(2, 1000))
y = np.array([])
for a in x :
    a = int(a)
    m_h = (2 * a) ** 50 + 1
    y = np.append(y, (sqrt(8 / a * (log(4 * m_h / 0.05)))))
plt.plot(x, y, label='(1)')

y = np.array([])
for a in x :
    a = int(a)
    m_h = (2 * a) ** 50 + 1
    y = np.append(y, sqrt(2 * log(2 * a * m_h) / a) + sqrt(2 / a * log(1 / 0.05)) + 1 / a)
plt.plot(x, y, label='(2)')

y = np.array([])
for a in x :
    a = int(a)
    m_h = (2 * a) ** 50 + 1
    y = np.append(y, sqrt(1 / a * log(6 * m_h / 0.05)))
plt.plot(x, y, label='(3)')

y = np.array([])
for a in x :
    a = int(a)
    m_h = (a ** 2) ** 50 + 1
    y = np.append(y, sqrt(1 / (2 * a) * log(4 * m_h / 0.05)))
plt.plot(x, y, label='(4)')


plt.legend()
plt.show()