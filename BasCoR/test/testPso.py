# 完成pso的测试
from Comp.Pso import Pos
import numpy as np

a = [-2.9, 4.2]
b = [12.00, 5.7]
a = np.array(a)
b = np.array(b)





if __name__ == '__main__':
    def func(args):
        fs = 21.5 + args[0] * np.sin(4 * np.pi * args[0]) + args[1] * np.sin(20 * np.pi * args[1])
        return fs
    p = Pos(a=a, b=b, func=func, Nn=32, position="max", c1=2.000, c2=2.000, w=0.04, adm=True, vmin=-3, vmax=12)
    p.fit(num=1000)
    print(np.max(p.ff))
