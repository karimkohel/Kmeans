import numpy as np

class Kmeans:
    
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iterations = max_iters
        self._init_clusters()

    def _init_clusters(self):
        self.clusters = [ [] for _ in range(self.K) ]
        self.centroids = []

    def fit(self, data):
        pass