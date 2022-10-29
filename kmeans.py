import numpy as np

class Kmeans:
    
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iterations = max_iters
        self._init_clusters()


    def fit(self, data):
        pass
