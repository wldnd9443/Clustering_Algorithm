# Clustering_Algorithms

## 소개
Clustering은 지도학습의 Classification과 다르게 비지도 학습입니다. Classification의 경우 데이터가 주어질 때 Label도 같이 주어진다면 Clustering은 별도의 Label이 사전에 미리 정의되지 않기 때문에 스스로 비슷한 속성을 가진 데이터를 묶어주는 역할을 해야합니다. 이 Repository에서는 여러 군집화 알고리즘(Clustering Algorithm)을 구현해보려 합니다. 임의로 데이터를 흩뿌리고 이 데이터들을 군집화 하는 과정을 Step 별로 나누어 어떻게 군집화가 진행되는지 그 과정을 볼 수 있도록 코드를 설계합니다.
 
### K-means Algorithm
K-Means 알고리즘은 N개의 데이터에서 K개의 Centroid(중심점)을 임의로 정하고 이 Centroid로 부터 데이터들의 거리의 총합이 가장 작은 값으로 수렴할 때 까지 수행을 반복하는 알고리즘입니다. 구체적으로 다음과 같은 단계를 따라 진행합니다.

1. K개의 임의의 중심점을 배치합니다.
2. 배치된 데이터가 가장 가까운 Centroid에 소속됩니다.
3. 소속된 데이터들의 Centroid를 구하여 갱신합니다.
4. 2번과 3번의 단계를 수렴할 때 까지 진행합니다.

### Mean Shift Algorithmn
Mean Shift 알고리즘은 영상에서 사물을 추적할 때도 쓰일 수 있는 유용한 알고리즘 입니다. K-means 알고리즘에서는 인접한 데이터의 거리의 총 합으로 알고리즘을 진행시켜나갔다면, Mean Shift는 정해진 반경을 설정하고 그 반경 내의 데이터가 가지는 Mean값으로 이동하면서 데이터의 분포의 중심으로 향하는 알고리즘입니다. 구체적으로 다음과 같은 단계를 따라 진행합니다.

1. 임의의 중심과 그 중심으로부터 일정한 반경을 설정합니다. 
2. 그 반경 안에 들어오는 데이터를 구합니다.
3. 이 데이터들의 무게중심의 좌표로 중심을 이동합니다.
4. 2번과 3번의 단계를 수렴할 때 까지 진행합니다.

## 구현 과정 

### Data Generation

2차원 공간에 직관적으로 4개의 군집을 이룰 수 있는 데이터를 배치합니다.
```
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

# X1MAX = -1
# X1MIN = 2
# X2MAX = -1
# X2MIN = 2
    
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
```
![data_generation](https://user-images.githubusercontent.com/44831709/134933721-3f3befc6-1e5b-4b8b-9ae7-075aa462ee9a.png)

### K-means Algorithmn

```
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
```

![k-means_result](https://user-images.githubusercontent.com/44831709/134936396-0eb2d763-7122-4450-b2ad-4e3575032e1d.png)

### Mean Shift Algorithmn
```
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
```

![mean_shift_result](https://user-images.githubusercontent.com/44831709/134941144-241c435c-0d87-4c2e-aeaa-3f9a20c0538c.png)

