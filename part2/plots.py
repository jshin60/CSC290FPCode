import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 1000, 1000)
y1 = 4*x
y2 = 2 + 10*(np.floor(x/8)) + 5*(np.floor((x%8)/4)) + 4*((x%8)%4)
plt.plot(x, y1, label = "unvectorized")
plt.plot(x, y2, label = "vectorized")
plt.xlabel("csrRowPtr_host[row + 1] - csrRowPtr_host[row]")       
plt.ylabel("number of size_t and SIMD operations per iteration")        
plt.title("Code Block 1 number of size_t and SIMD operations")
plt.legend()
plt.savefig('block1.png')
plt.close()

b2x = np.floor(x/16)
y3 = 4 * b2x
y4 = 2 + 24*(np.floor(b2x/8)) + 11*(np.floor((b2x%8)/4)) + 4*((b2x%8)%4)
plt.plot(x, y3, label = "unvectorized")
plt.plot(x, y4, label = "vectorized")
plt.xlabel("Z")       
plt.ylabel("number of size_t and SIMD operations per iteration")        
plt.title("Code Block 2 number of size_t and SIMD operations")
plt.legend()
plt.savefig('block2.png')
plt.close()

x = np.linspace(1, 1000, 1000)
y1 = 11*x
y2 = 5 + 30*(np.floor(x/8)) + 15*(np.floor((x%8)/4)) + 11*((x%8)%4)
plt.plot(x, y1, label = "unvectorized")
plt.plot(x, y2, label = "vectorized")
plt.xlabel("csrRowPtr_host[row + 1] - csrRowPtr_host[row]")       
plt.ylabel("number of size_t and SIMD operations per iteration")        
plt.title("Code Block 3 number of size_t and SIMD operations")
plt.legend()
plt.savefig('block3.png')
plt.close()