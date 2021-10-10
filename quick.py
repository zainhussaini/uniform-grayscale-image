#!/usr/bin/env python3

import numpy as np

a = np.arange(0, 10).reshape(2, 5)
print(a)
b = a[:, a[0, :]%2 == 0]
print(b)
b[0, 0] = 100
print(b)
print(a)
