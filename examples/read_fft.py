#%%
import serialem
import numpy as np
import matplotlib.pyplot as plt

serialem.Echo("Hi")
data = np.asarray(serialem.bufferImage("AF"))
plt.imshow(data)
plt.show()
x,y = data.shape
data_ft = np.fft.fftshift(np.fft.fft2(data))
data_br = 
data = np.fft.ifft2(np.fft.ifftshift(data)).real
data = data[0]
plt.imshow(data)
plt.show()

sx, sy = data.shape
X, Y = np.ogrid[0:sx, 0:sy]


R = np.hypot(X - sx/2, Y - sy/2)

rad = np.arange(1, np.max(R), 1)
intensity = np.zeros(len(rad))
index = 0
bin_size = 1
for i in rad:
  mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
  values = data[mask]
  intensity[index] = np.mean(values)
  index += 1

plt.plot(intensity)
plt.show()