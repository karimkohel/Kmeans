import enum
from json.tool import main
from unicodedata import name
import numpy as np

np.random.seed(78)




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
            oldCentroids = self.centroids
            # update data clusters
            self.clusters = self._create_clusters(oldCentroids)
            # update centroids
            newCentroids = self._get_new_centroids(self.clusters)
            # check error
            if self._is_converged(oldCentroids, newCentroids):
                break
        # return cluster labels
        return self._get_cluster_labels()

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cIndx, cluster in enumerate(clusters):
            for sampleIndx in cluster:
                labels[sampleIndx] = cIndx

        return labels

    def _is_converged(self, oldCentroids, newCentroids):
        distances = [self.euclidean_distance(oldCentroids[i], newCentroids[i]) for i in range(self.K)]
        return (sum(distances) == 0)

    def _get_new_centroids(self, clusters):
        _centroids = np.zeros(self.K, self.n_features)
        for cIndx, cluster in enumerate(clusters):
            mean = np.mean(self.data[cluster], axis=0)
            _centroids[cIndx] = mean

        return _centroids



    def _create_clusters(self, centroids):
        _clusters = [[] for _ in range(self.K)]

        for indx, sample in enumerate(self.data):
            cendroid_indx = self._closest_centroid(sample, centroids)
            _clusters[cendroid_indx].append(indx)
        return _clusters

    def _closest_centroid(self, sample, centroids):
        # will return the list index of the closest centroid from the centroids list
        distances = [self.euclidean_distance(sample, centroid) for centroid in centroids]
        return np.argmin(distances)

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1-p2))**2)

if __name__ == '__main__':
    print("Hello")