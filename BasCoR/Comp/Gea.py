import numpy as np
import pandas as pd
pd.set_option('precision', 5)
class GEA:
    # 基因的长度
    N_ = []
    c_N = []
    new_N = []
    s = []

    def __init__(self, size, D1, D2, X2, f, k=None, Np=None, jingdu=0.000001, point="max"):
        """
        :param size:"种群的规模"
        :param D:"染色体的取值范围"
        :param point:"目标是求最大值还是最小值，默认是求最大值"
        :param f:"目标函数"
        :param :X2"目标函数的常数"
        """
        self.D1 = D1
        self.D2 = D2
        self.size = size
        self.jingdu = jingdu
        self.f = f
        self.point = point
        self.X2 = X2
        self.k = int(np.log2((D2.max() - D1.min()) * (1 / jingdu)))
        self.N_ = []
        self.c_N = []
        self.new_N = []
        self.s = []
        n = []
        N = []
        if len(Np) == 0:
            D = np.column_stack((D1, D2))
            for i in range(0, size):
                for i in range(0, len(D)):
                    x = np.random.uniform(D[i][0], D[i][1])  #
                    n.append(x)
                N.append(np.array(n))
                n = []
            self.N_ = np.array(N)
        else:
            self.N_ = Np

    def ToTen(self, N):
        k = self.k  # 编码长度
        D1 = self.D1
        D2 = self.D2
        t_N = N * (D2 - D1) / (2 ** k - 1) + D1

        return t_N

    def ToCode(self, N):
        k = self.k  # 编码长度
        D1 = self.D1
        D2 = self.D2
        c_N = (N - D1) * ((2 ** k) - 1) / (D2 - D1)
        c_N = c_N.astype(np.int32)
        return c_N

    def computP(self, f, N):  # 计算各种指标
        f = self.f
        X2 = self.X2
        F = f(N, X2)
        P = F / (F.sum())
        return P

    def fit(self, num=100, lv=0.7, vlv=0.01, tl=0.1):  # 编写训练接口
        N = self.N_
        f = self.f
        k = self.k
        point = self.point
        X2 = self.X2
        self.new_N.extend(N)
        if point == "max":
            for i in range(0, num):
                c_N = self.ToCode(N)
                for g in range(0, 50):
                    t1 = np.random.randint(0, N.shape[0])
                    t2 = np.random.randint(0, N.shape[0])
                    t3 = np.random.randint(0, N.shape[0])
                    t4 = np.random.randint(0, N.shape[0])
                    t5 = np.random.randint(0, N.shape[0])
                    t6 = np.random.randint(0, N.shape[0])
                    t7 = np.random.randint(0, N.shape[0])
                    t8 = np.random.randint(0, N.shape[0])
                    t9 = np.random.randint(0, N.shape[0])
                    t10 = np.random.randint(0, N.shape[0])
                    t11 = np.random.randint(0, N.shape[0])
                    t12 = np.random.randint(0, N.shape[0])
                    t13 = np.random.randint(0, N.shape[0])
                    t14 = np.random.randint(0, N.shape[0])
                    t15 = np.random.randint(0, N.shape[0])
                    t16 = np.random.randint(0, N.shape[0])

                    f1 = c_N[t1]
                    f2 = c_N[t2]
                    f3 = c_N[t3]
                    f4 = c_N[t4]
                    f5 = c_N[t5]
                    f6 = c_N[t6]
                    f7 = c_N[t7]
                    f8 = c_N[t8]
                    f9 = c_N[t9]
                    f10 = c_N[t10]
                    f11 = c_N[t11]
                    f12 = c_N[t12]
                    f13 = c_N[t13]
                    f14 = c_N[t14]
                    f15 = c_N[t15]
                    f16 = c_N[t16]

                    f1 = f1.astype(np.int32)
                    f2 = f2.astype(np.int32)
                    f3 = f3.astype(np.int32)
                    f4 = f4.astype(np.int32)
                    f5 = f5.astype(np.int32)
                    f6 = f6.astype(np.int32)
                    f7 = f7.astype(np.int32)
                    f8 = f8.astype(np.int32)
                    f9 = f9.astype(np.int32)
                    f10 = f10.astype(np.int32)
                    f11 = f11.astype(np.int32)
                    f12 = f12.astype(np.int32)
                    f13 = f13.astype(np.int32)
                    f14 = f14.astype(np.int32)
                    f15 = f15.astype(np.int32)
                    f16 = f16.astype(np.int32)

                    ##1
                    e1 = np.random.randint(1, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f1 & e2) | (f2 & e1)
                    s2 = (f2 & e2) | (f1 & e1)

                    b = np.random.randint(1, 16)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##2
                    e1 = np.random.randint((2 ** k) / 4, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f3 & e2) | (f4 & e1)
                    s2 = (f4 & e2) | (f3 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##3                  
                    e1 = np.random.randint((2 ** k) / 16, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f5 & e2) | (f6 & e1)
                    s2 = (f6 & e2) | (f5 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                        ##4
                    e1 = np.random.randint((2 ** k) / 32, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f7 & e2) | (f8 & e1)
                    s2 = (f8 & e2) | (f7 & e1)

                    b = np.random.randint(3, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##5                  
                    e1 = np.random.randint((2 ** k) / 64, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f9 & e2) | (f10 & e1)
                    s2 = (f10 & e2) | (f9 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##6
                    e1 = np.random.randint((2 ** k) / 128, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f11 & e2) | (f12 & e1)
                    s2 = (f12 & e2) | (f11 & e1)

                    b = np.random.randint(7, 51)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ## 7
                    e1 = np.random.randint(0, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f13 & e2) | (f14 & e1)
                    s2 = (f14 & e2) | (f13 & e1)

                    b = np.random.randint(1, 512)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##8
                    e1 = np.random.randint(7689, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f15 & e2) | (f16 & e1)
                    s2 = (f16 & e2) | (f15 & e1)

                    b = np.random.randint(1, 516)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)
                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) > f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                N = self.new_N
                N = np.array(N)
                F = f(N, X2)
                TT = pd.DataFrame(N)
                TT["F"] = F
                TT = TT.sort_values("F")
                TT.index = range(TT.shape[0])
                L = TT.shape[0] - 30
                N = TT.iloc[L:, 0:-1].values
                self.new_N = []
                TT = []
                self.new_N.extend(N)
            return np.array(self.new_N),self.f(np.array(self.new_N),self.X2).max()

        if point == "min":

            for i in range(0, num):
                c_N = self.ToCode(N)
                for g in range(0, 2):
                    t1 = np.random.randint(0, N.shape[0])
                    t2 = np.random.randint(0, N.shape[0])
                    t3 = np.random.randint(0, N.shape[0])
                    t4 = np.random.randint(0, N.shape[0])
                    t5 = np.random.randint(0, N.shape[0])
                    t6 = np.random.randint(0, N.shape[0])
                    t7 = np.random.randint(0, N.shape[0])
                    t8 = np.random.randint(0, N.shape[0])
                    t9 = np.random.randint(0, N.shape[0])
                    t10 = np.random.randint(0, N.shape[0])
                    t11 = np.random.randint(0, N.shape[0])
                    t12 = np.random.randint(0, N.shape[0])
                    t13 = np.random.randint(0, N.shape[0])
                    t14 = np.random.randint(0, N.shape[0])
                    t15 = np.random.randint(0, N.shape[0])
                    t16 = np.random.randint(0, N.shape[0])

                    f1 = c_N[t1]
                    f2 = c_N[t2]
                    f3 = c_N[t3]
                    f4 = c_N[t4]
                    f5 = c_N[t5]
                    f6 = c_N[t6]
                    f7 = c_N[t7]
                    f8 = c_N[t8]
                    f9 = c_N[t9]
                    f10 = c_N[t10]
                    f11 = c_N[t11]
                    f12 = c_N[t12]
                    f13 = c_N[t13]
                    f14 = c_N[t14]
                    f15 = c_N[t15]
                    f16 = c_N[t16]

                    f1 = f1.astype(np.int32)
                    f2 = f2.astype(np.int32)
                    f3 = f3.astype(np.int32)
                    f4 = f4.astype(np.int32)
                    f5 = f5.astype(np.int32)
                    f6 = f6.astype(np.int32)
                    f7 = f7.astype(np.int32)
                    f8 = f8.astype(np.int32)
                    f9 = f9.astype(np.int32)
                    f10 = f10.astype(np.int32)
                    f11 = f11.astype(np.int32)
                    f12 = f12.astype(np.int32)
                    f13 = f13.astype(np.int32)
                    f14 = f14.astype(np.int32)
                    f15 = f15.astype(np.int32)
                    f16 = f16.astype(np.int32)

                    ##1
                    e1 = np.random.randint(1, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f1 & e2) | (f2 & e1)
                    s2 = (f2 & e2) | (f1 & e1)

                    b = np.random.randint(1, 16)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##2
                    e1 = np.random.randint((2 ** k) / 4, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f3 & e2) | (f4 & e1)
                    s2 = (f4 & e2) | (f3 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##3                  
                    e1 = np.random.randint((2 ** k) / 16, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f5 & e2) | (f6 & e1)
                    s2 = (f6 & e2) | (f5 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                        ##4
                    e1 = np.random.randint((2 ** k) / 32, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f7 & e2) | (f8 & e1)
                    s2 = (f8 & e2) | (f7 & e1)

                    b = np.random.randint(3, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##5                  
                    e1 = np.random.randint((2 ** k) / 64, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f9 & e2) | (f10 & e1)
                    s2 = (f10 & e2) | (f9 & e1)

                    b = np.random.randint(31, 511)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##6
                    e1 = np.random.randint((2 ** k) / 128, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f11 & e2) | (f12 & e1)
                    s2 = (f12 & e2) | (f11 & e1)

                    b = np.random.randint(7, 51)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ## 7
                    e1 = np.random.randint(0, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f13 & e2) | (f14 & e1)
                    s2 = (f14 & e2) | (f13 & e1)

                    b = np.random.randint(1, 2)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)

                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                    ##8
                    e1 = np.random.randint(7689, 2 ** k - 1)
                    e2 = e1 ^ (2 ** k - 1)
                    s1 = (f15 & e2) | (f16 & e1)
                    s2 = (f16 & e2) | (f15 & e1)

                    b = np.random.randint(1, 516)
                    s1 = s1 ^ b
                    s2 = s2 ^ b

                    s1 = self.ToTen(s1)
                    s2 = self.ToTen(s2)
                    sf1 = s1.reshape(1, 2)
                    sf2 = s2.reshape(1, 2)

                    if f(sf1, X2) < f(sf2, X2):
                        self.new_N.append(s1)
                    else:
                        self.new_N.append(s2)

                N = self.new_N
                N = np.array(N)
                F = f(N, X2)
                TT = pd.DataFrame(N)
                TT["F"] = F
                TT = TT.sort_values("F")
                TT = TT[::-1]
                TT.index = range(TT.shape[0])
                L = TT.shape[0] - 4500
                N = TT.iloc[L:, 0:-1].values
                self.new_N = []
                TT = []
                self.new_N.extend(N)
            return np.array(self.new_N),self.f(np.array(self.new_N),self.X2).min()

    def getN(self):
        return np.array(self.N_)