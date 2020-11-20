from matplotlib import pyplot  as plt

x = range(4)
y = [15,13,14.5,17,20,25,26,26,27,22,18,15]
plt.plot(x, y)
plt.savefig('test.png')