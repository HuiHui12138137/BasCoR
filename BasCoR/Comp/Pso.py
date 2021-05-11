import numpy as np
# 打包优化后的代码：用矩阵运算代替for循环，进一步提高运行速度
class Pos():
    N = []  # 种群
    x = []   #位置
    pbest = [] #个体历史最优
    gbest = [] #群体历史最优
    v = []  #速度
    g = []   #所有历史最优
    ff = []   #所有适应值

    def __init__(self, a, b, func, Nn, position="max", c1=2.0, c2=2.0, w=0.3, adm=False, vmin=1, vmax=5):
        self.a = a
        self.b = b
        self.func = func
        self.position = position
        self.Nn = Nn
        self.vmin = vmin
        self.vmax = vmax
        self.c1 = c1
        self.c2 = c2
        self.adm = adm
        self.w = w
        self.N = []
        self.x = []
        self.pbest = []
        self.gbest = []
        self.v = []
        self.g = []
        self.ff = []
        self.Toinit()

    def Toinit(self):
        Nn = self.Nn
        a = self.a
        b = self.b
        x = self.x
        N = self.N
        func=self.func
        pbest = self.pbest
        position = self.position
        vv = []
        for i in range(0, Nn):
            for i in range(0, len(a)):
                x.append(np.random.uniform(a[i], b[i]))
                vv.append(np.random.randint(self.vmin, self.vmax))
            N.append(np.array(x))
            self.v.append(np.array(vv))
            x = []
            vv = []
            pbest = np.array(N)
            f0 = func(pbest.T)
            if position == "max":
                self.gbest = pbest[np.argmax(f0)]
            if position == "min":
                self.gbest = pbest[np.argmin(f0)]
            self.pbest = pbest
        self.v = np.array(self.v)
        return f0

    def overstep(self, N):  # 越界处理
        a = self.a  # 下界
        b = self.b  # 上界   #位置集合
        bb = np.array(N).copy()
        bb = np.array(N).copy()
        for i in range(0, len(a)):
            bb[bb[:, i] < a[i]] = a[i]
        for i in range(0, len(b)):
            bb[bb[:, i] > b[i]] = b[i]
        return bb

    def fit(self, num=1000):

        c1 = self.c1
        c2 = self.c2
        w = self.w
        x = self.N
        x = self.overstep(x)
        v = self.v
        pbest = np.array(self.pbest)
        gbest = np.array(self.gbest)
        position = self.position
        # 越界判断：
        x = self.overstep(x)
        for i in range(0, num):
            if self.adm == True:
                if 0 < i and i < int(num * 0.3):
                    w = 0.7
                if int(num * 0.3) < i and i < int(num * 0.7):
                    w = 0.4
                else:
                    w = 0.3
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
            x = x + v
            x = self.overstep(x)
            pf = self.func(pbest.T)
            f = self.func(x.T)
            for i in range(0, len(f)):
                if pf[i] < f[i]:
                    pbest[i] = x[i]
            gbest = pbest[np.argmax(pf)]
            self.g.append(gbest)
            self.ff.append(self.func(gbest))
