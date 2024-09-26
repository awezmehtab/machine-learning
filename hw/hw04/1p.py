import numpy as np

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    y = (2 * x + 3)**2
    return float(y[0,0])

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0][0]); y = float(v[1][0])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0][0]); y = float(v[1][0])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def gd(f, df, x0, step_size_fn, max_iter):
  xs = [x0]
  fs = [f(x0)]
  x_i = x0.copy()
  for t in range(max_iter):
    x_i = x_i - step_size_fn(x_i)*df(x_i)
    xs.append(x_i.copy())
    fs.append(f(x_i.copy()))
  return (x_i, fs, xs)

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

# Test case 1
ans=package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))
print(ans)

# Test case 2
ans=package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))