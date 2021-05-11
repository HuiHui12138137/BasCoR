from Comp.Gea import GEA
import numpy as np
def func(D,x2):
    args=np.array(D).T
    fs=21.5+args[0]*np.sin(4*np.pi*args[0])+args[1]*np.sin(20*np.pi*args[1])+x2
    return fs
if __name__ == '__main__':

    a=[-2.9,4.2]
    b=[12,5.7]
    X2=0.0
    p=GEA(size=400,D1=np.array(a),D2=np.array(b),X2=X2,f=func,k=None,Np=[],jingdu=0.001,point="min")
    gen_position,K=p.fit(num=1)
    print (K)