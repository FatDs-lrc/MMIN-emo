import multiprocessing

def func(x):
    return x*x

p = multiprocessing.Pool(4)
ans = p.map(func, range(100))
print(ans)
