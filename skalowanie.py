import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data


# kernele

def h1(t):
    return np.where((t >= 0) & (t < 1), 1.0, 0.0)

def h2(t):
    return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)

def h3(t):
    return np.where(t == 0, 1.0, np.sin(t) / t)


# MSE
def MSE(y_interp, y_true):
    return np.mean((y_interp - y_true) ** 2)


# Pomniejszanie
def zmniejszanie(img, factor):
    m, n = img.shape
    new_m, new_n = int(m / factor), int(n / factor)
    new_img = np.zeros((new_m, new_n))

    for i in range(new_m):
        for j in range(new_n):
            block = img[
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ]
            new_img[i, j] = np.mean(block)

    return new_img


# Powiększanie – interpolacja

def powiekszanie(img, scale, kernel):
    m, n = img.shape
    new_m, new_n = int(m * scale), int(n * scale)

    # pusty obraz
    new_img = np.zeros((new_m, new_n))

    # siatka wspolrzędnych
    h = 1
    kernel_size = 7

    # kernel w osi x i y
    x = np.arange(-3, 4)
    X, Y = np.meshgrid(x, x)
    kernel_2d = kernel(X / h) * kernel(Y / h)

    # normalizacja kernela
    kernel_2d /= np.sum(kernel_2d)

    for i in range(m):
        for j in range(n):
            new_img[int(i*scale), int(j*scale)] = img[i, j]

    new_img = convolve2d(new_img, kernel_2d, mode='same')

    return new_img


# obraz
img = data.camera()
img = img.astype(float) / 255.0

# Skalowanie
img_small = zmniejszanie(img, factor=2)
img_big = powiekszanie(img_small, scale=2, kernel=h3)

mse_val = MSE(img, img_big)

print("MSE:", mse_val)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Oryginalny obraz')

plt.subplot(1,3,2)
plt.imshow(img_small, cmap='gray')
plt.title('Pomniejszony')

plt.subplot(1,3,3)
plt.imshow(img_big, cmap='gray')
plt.title('Po powiększeniu')

plt.show()
