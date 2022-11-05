from json.tool import main
from unicodedata import name
import numpy as np

np.random.seed(78)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2))**2)



class Kmeans:
    
    def __init__(self, K=2, max_iters=100):
        self.K = K
        self.max_iterations = max_iters
        self._init_clusters()

    def _init_clusters(self):
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []


    def predict(self, data):
        self.data = data
        self.n_samples, self.n_features = data.shape

        # init random centroids for start
        randomCentroids = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.data[indx] for indx in randomCentroids]

        # optimize centroids location
        for _ in range(self.max_iterations):
            # update data clusters
            # update centroids
            # check error

        # return cluster labels

    def _create_clusters(self, centroids):
        _clusters = [[] for _ in range(self.K)]

        for indx, sample in enumerate(self.data):
            cendroid_indx = self._closest_centroid()
            pass

    def _closest_centroid(self, point, centroids):
        pass


if __name__ == '__main__':
    print("Hello")