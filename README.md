# Clustering_Algorithms

## 소개
  Clustering은 지도학습의 Classification과 다르게 비지도 학습입니다. Classification의 경우 데이터가 주어질 때 Label도 같이 주어진다면 Clustering은 별도의 Label이 사전에 미리 정의되지 않기 때문에 스스로 비슷한 속성을 가진 데이터를 묶어주는 역할을 해야합니다. 이 Repository에서는 여러 군집화 알고리즘(Clustering Algorithm)을 구현해보려 합니다. 임의로 데이터를 흩뿌리고 이 데이터들을 군집화 하는 과정을 Step 별로 나누어 어떻게 군집화가 진행되는지 그 과정을 볼 수 있도록 코드를 설계합니다.
 
### K-means Algorithm
K-Means 알고리즘은 N개의 데이터에서 K개의 Centroid(중심점)을 임의로 정하고 이 Centroid로 부터 데이터들의 거리가 가장 작은 값으로 수렴할 때 까지 수행을 반복하는 알고리즘입니다. 구체적으로 다음과 같은 Step을 밟습니다.

1. K개의 임의의 중심점을 배치합니다.
2. 배치된 데이터가 가장 가까운 Centroid에 소속됩니다.
3. 소속된 데이터들의 Centroid를 구하여 갱신합니다.
4. 2번과 3번의 단계를 수렴할 때 까지 진행합니다.

### Mean Shift Algorithmn
