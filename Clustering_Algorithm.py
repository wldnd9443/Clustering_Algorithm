import numpy as np
import matplotlib.pyplot as plt

# Data Generation
NOISE = 0.02
mat_covs = np.array([[[4,2.5],[2.5,4]],[[1,0],[0,4]],[[1,0],[0,1]],[[1,0],[0,1]]])*NOISE

mus =  np.array([[1,1],[0,0],[1.5,0.5],[-.5,1.2]])
Ns = np.array([400,400,400,400])
clss = [1,2,3,4]

X = np.zeros((0,mus.shape[1]))
Y = np.zeros(0)

for mu, mat_cov, N, cls in zip(mus, mat_covs, Ns, clss):
    X_ = np.random.multivariate_normal(mu, mat_cov, N)
    Y_ = np.ones(N)*cls
    X = np.vstack((X,X_))
    Y = np.hstack((Y,Y_))
    
X1MAX = max(X[:,0])
X1MIN = min(X[:,0])
X2MAX = max(X[:,1]) 
X2MIN = min(X[:,1])
    
def plot_data(X,Y, mus=None):
    cls_unique = np.unique(Y)
    legends = []
    for cls in cls_unique:
        idx = Y==cls
        plt.plot(X[idx,0],X[idx,1],'.')
        legends.append(cls.astype('int'))
        
    if mus is not None:
        plt.plot(mus[:,0],mus[:,1],'kx')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(legends)
    plt.grid(True)
    plt.xlim([X1MIN,X1MAX])
    plt.ylim([X2MIN,X2MAX])
    plt.show()
    
plot_data(X,Y)


## K-Means Algorithm
N_iter = 5
k = 4

np.random.seed(5)
mus1 = np.random.uniform(size = (k,1), low = X1MIN, high = X1MAX)
mus2 = np.random.uniform(size = (k,1), low = X2MIN, high = X2MAX)
mus = np.hstack((mus1,mus2))

clss = np.argmin(np.sum((X[:,None,:]-mus[None,:,:])**2,axis=2),axis=1) # assignment step
plot_data(X,clss,mus=mus)

for it in range(N_iter):
    clss = np.argmin(np.sum((X[:,None,:]-mus[None,:,:])**2,axis=2),axis=1) # assignment step
    uq_cls = np.arange(k)
    for k,cls in enumerate(uq_cls):
        I = clss==cls
        mus[k,:] = X[I,:].mean(axis=0) # measurement step
    plot_data(X,clss,mus=mus)

## Mean Shift Algorithm
N_iter = 5
N_frame  = 3
k = 10

I_frame = I_frame = np.floor(np.arange(1,N_frame+1)/N_frame*N_iter).astype('int')-1

eps = 0.5

np.random.seed(3)

mus = X[np.random.permutation(X.shape[0])[:k],:]


clss = np.argmin(np.sum((X[:,None,:]-mus[None,:,:])**2,axis=2),axis=1)
plot_data(X,clss,mus=mus)
for it in range(N_iter):
    uq_cls = np.arange(k)
    for k,cls in enumerate(uq_cls):
        dist = np.sqrt(np.sum((X-mus[k,:])**2,axis=1))
        I = dist < eps
        if I.size>0:
            mus[k,:]=X[I,:].mean(axis=0)
            
    clss = np.argmin(dist_cls,axis=1)
    dist_mus = np.sum((mus[:,None,:]-mus[None,:,:])**2,axis=2)
    matches = np.arange(mus.shape[0])
    for match in matches:
        I = dist_mus[match,:] < 0.01
        matches[I] = match

    if it in I_frame:
        plot_data(X,matches[clss],mus=mus)
