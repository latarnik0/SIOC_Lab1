import numpy as np
import matplotlib.pyplot as plt

N = 100
# Równomiernie
x = np.linspace(-np.pi, np.pi, N)

# Losowane z przedziału
#x = np.random.uniform(-np.pi, np.pi, N)
#x = np.sort(x)

# Punkty gęsto przy 0 i mało przy krańcach
#u = np.linspace(-1, 1, N)      # pomocnicza siatka równomierna
#x = np.pi * u**3

# Losowane z rozkładu normalnego
# x = np.random.normal(loc=0, scale=1, size=N)

f1 = np.sin(x)
f2 = np.sin(1/x)
f3 = np.sign(np.sin(8*x))

# Nowe punkty
x_new = np.linspace(-np.pi, np.pi, 10*N)

# Kernele
def h3(t):
    return np.where(t == 0, 1.0, np.sin(t) / t)


def h2(t):
    return np.where((t >= -0.5) & (t < 0.5), 1, 0)


def h1(t):
    return np.where((t >= 0) & (t < 1), 1, 0)


# Interpolacja
h = x[1] - x[0]
y_new = np.zeros_like(x_new)

for i, xi in enumerate(x_new):
    v = (xi - x) / h
    w = h3(v)

    if np.sum(w) == 0:
        idx = np.argmin(np.abs(x - xi))
        y_new[i] = f3[idx]
    else:
        y_new[i] = np.sum(w * f3) / np.sum(w)

# MSE
def MSE(y_interp, y_true):
    mse_val = np.mean((y_interp - y_true)**2)
    return mse_val
# funkcje zageszczone do MSE 
f1_dense = np.sin(x_new)
f2_dense = np.sin(1/x_new)
f3_dense = np.sign(np.sin(8*x_new))

mse_val = MSE(y_new, f3_dense)

plt.plot(x, f3, 'o', label='Oryginalne punkty N=100')
plt.plot(x_new, y_new, '-', label='Funkcja interpolowana N=1000')
plt.title('Interpolacja f2 z h3')
plt.legend()
plt.grid(True)
plt.show()
print(mse_val)
