import matplotlib.pyplot as plt

x = [6, 10, 20, 60, 100, 150]
y = [8.075, 8.368, 8.620, 8.756, 8.816, 8.827]
x2 = [15, 20, 30, 35, 40]
y2 = [6.222, 8.368, 8.535, 8.649, 8.720]

x3 = [670, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 2400]
y3 = [7.553, 8.466, 8.532, 8.581, 8.608, 8.620, 5.662, 2.830, 1.943, 1.019]

fig = plt.figure()
fig.add_subplot(3,1,1)
plt.plot(x, y, 'bo', markersize=5)
plt.plot(x, y)
plt.xlabel('M')
plt.ylabel('Speedup')
plt.title('Speedup of BenchMax as a Function of M (N=20)')
plt.grid()
fig.add_subplot(3,1,2)
plt.plot(x2, y2, 'bo', markersize=5)
plt.plot(x2, y2)
plt.xlabel('N')
plt.ylabel('Speedup')
plt.title('Speedup of BenchMax as a Function of N (M=10)')
plt.grid()
fig.add_subplot(3,1,3)
plt.plot(x3, y3, 'bo', markersize=5)
plt.plot(x3, y3)
plt.xlabel('Number of Declarations')
plt.ylabel('Speedup')
plt.title('Speedup of BenchMax as a Function of #(k declarations) (M = N = 20)')
plt.grid()
plt.tight_layout()

# Show plot
plt.show()