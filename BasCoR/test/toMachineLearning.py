import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.datasets import make_blobs  # 制作几个簇
import matplotlib.pyplot as plt
import warnings
from Comp.Pso import Pos
import numpy as np

warnings.filterwarnings("ignore")
a = [2, 1, 100, 2, 0]
b = [100, 40, 2000, 5, 1]
a = np.array(a)
b = np.array(b)

X, y = make_blobs(n_samples=7000, n_features=40, centers=40, random_state=2)  # 7000条数据，40个簇40个中心


def func(A, X2):
    init_to = ["k-means++", "random"]
    N_x = A.T
    num = len(N_x)
    N_x = np.array(N_x).astype(np.int32)
    X = X2
    fs = []
    if len(N_x.shape) > 1:
        for i in range(0, num):
            clusterer = KMeans(n_clusters=N_x[i][0], n_init=N_x[i][1], max_iter=N_x[i][2], n_jobs=N_x[i][3],
                               init=init_to[N_x[i][4]], random_state=10).fit(X)
            cluster_labels = clusterer.labels_
            silhouette_avg = silhouette_score(X, cluster_labels)
            fs.append(silhouette_avg)
    else:
        clusterer = KMeans(n_clusters=N_x[0], n_init=N_x[1], max_iter=N_x[2], n_jobs=N_x[3], init=init_to[N_x[4]],
                           random_state=10).fit(X)
        cluster_labels = clusterer.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        fs.append(silhouette_avg)
    return fs


p = Pos(a=a, b=b, X2=X, func=func, Nn=5, position="max", c1=2.000, c2=2.000, w=0.04, adm=True, vmin=-3, vmax=12)
p.fit(num=3)
print(p.ff)