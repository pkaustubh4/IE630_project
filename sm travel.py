import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from statsmodels.distributions.empirical_distribution import ECDF

reg = [87, 165, 236, 323, 277, 440, 269, 342, 175, 273, 115, 56]
crd = [89, 243, 221, 180, 301, 490, 394, 347, 240, 269, 145, 69]
z = [a+b for a, b in zip(reg, crd)]

mu = np.mean(z)
sigma = np.sqrt(np.var(z))

# ecdf = ECDF(z)
# plt.plot(ecdf.x, ecdf.y)
# plt.show()

print("Ho: z is IID N[478.83, 218.02]." + '\n' + "Running K-S test..." + '\n')
N = 12

def f(x):
  a = [sc.norm.cdf(p, mu, sigma) for p in x]
  return a

f_x = f(z)
f_x_sorted = np.sort(f_x)
 
#Calculating max(i/N-Ri)
plus_max = list()
for i in range(1, N + 1):
  j = i/N - f_x_sorted[i-1]
  plus_max.append(j)
K_plus_max = np.max(plus_max)

#Calculating max(Ri-((i-1)/N))
minus_max = list()
for i in range(0, N):
  y = f_x_sorted[i]-i/N
  minus_max.append(y)
K_minus_max = np.max(minus_max)

r = np.arange(120, 935, 1)
fig = plt.figure(figsize = (7, 5))
plt.plot(r, f(r), color="orange")

f_z_hat = [] #Calculating emperical CDF
for i in range(N):
  a = [c for c in z if c<=z[i]]
  f_z_hat.append(len(a)/N)
plt.scatter(z, f_z_hat, marker='.')

plt.legend(["Theoretical", "Emperical"])
plt.ylabel("F(z)")
plt.xlabel("z")

plt.show()
 
#Calculating KS Statistic
K_max = max(K_plus_max, K_minus_max)
K_alpha = 1.35810/np.sqrt(N)
if K_max<K_alpha: #Running the test
  print("Test failed. Ho NOT rejected." + '\n' + "z(i) is IID N[478.83, 218.02].")
else:
  print("Test passed. Ho rejected." + '\n' + "z(i) is not IID N[478.83, 218.02].")