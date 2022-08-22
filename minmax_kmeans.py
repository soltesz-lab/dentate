#!/usr/bin/env python3.5
# coding: utf-8
# Implementation of KMeans clustering with minimum cluster size constraint (Bradley et al., 2000).
# Based on code from https://github.com/Behrouz-Babaki/MinSizeKmeans


import pulp
import random

def l2_distance(point1, point2):
    return sum([(float(i)-float(j))**2 for (i,j) in zip(point1, point2)])

class subproblem(object):
    def __init__(self, centroids, data, min_size, max_size):

        k = len(centroids)
        
        self.centroids = centroids
        self.data = data
        self.min_size = min_size
        self.max_size= max_size
        self.n = len(data)
        self.k = k

        self.create_model()

    def create_model(self):
        def distances(assignment):
            return l2_distance(self.data[assignment[0]], self.centroids[assignment[1]])

        clusters = list(range(self.k))
        assignments = [(i, j)for i in range(self.n) for j in range(self.k)]

        # outflow variables for data nodes
        self.y = pulp.LpVariable.dicts('data-to-cluster assignments',
                                  assignments,
                                  lowBound=0,
                                  upBound=1,
                                  cat=pulp.LpInteger)

        # outflow variables for cluster nodes
        self.b = pulp.LpVariable.dicts('cluster outflows',
                                  clusters,
                                  lowBound=0,
                                  upBound=self.n-self.min_size,
                                  cat=pulp.LpContinuous)

        # create the model
        self.model = pulp.LpProblem("Model for assignment subproblem", pulp.LpMinimize)

        # objective function
        self.model += pulp.lpSum([distances(assignment) * self.y[assignment] for assignment in assignments])

        # flow balance constraints for data nodes
        for i in range(self.n):
            self.model += pulp.lpSum(self.y[(i, j)] for j in range(self.k)) == 1

        # flow balance constraints for cluster nodes
        for j in range(self.k):
            self.model += pulp.lpSum(self.y[(i, j)] for i in range(self.n)) - self.min_size == self.b[j]
            
        # capacity constraint on outflow of cluster nodes
        for j in range(self.k):
            self.model += self.b[j] <= self.max_size - self.min_size

        # flow balance constraint for the sink node
        self.model += pulp.lpSum(self.b[j] for j in range(self.k)) == self.n - (self.k * self.min_size)


    def solve(self):
        self.status = self.model.solve()
        clusters = None
        if self.status == 1:
            clusters= [-1 for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.k):
                    if self.y[(i, j)].value() > 0:
                        clusters[i] = j
        return clusters

def initialize_centers(dataset, k):
    ids = list(range(len(dataset)))
    random.shuffle(ids)
    return [dataset[id] for id in ids[:k]]

def compute_centers(clusters, dataset):
    # canonical labeling of clusters
    ids = list(set(clusters))
    c_to_id = dict()
    for j, c in enumerate(ids):
        c_to_id[c] = j
    for j, c in enumerate(clusters):
        clusters[j] = c_to_id[c]

    k = len(ids)
    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]
    counts = [0] * k
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1
    for j in range(k):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])
    return clusters, centers

def minsize_kmeans(dataset, k, max_n_iter=5, min_size=0, max_size=None):
    n = len(dataset)
    if max_size == None:
        max_size = n

    centers = initialize_centers(dataset, k)
    clusters = [-1] * n

    it = 0
    converged = False
    while not converged and (it < max_n_iter):
        m = subproblem(centers, dataset, min_size, max_size)
        clusters_ = m.solve()
        if not clusters_:
            return None, None
        clusters_, centers = compute_centers(clusters_, dataset)

        converged = True
        i = 0
        while converged and i < len(dataset):
            if clusters[i] != clusters_[i]:
                converged = False
            i += 1
        clusters = clusters_
        it += 1

    return clusters, centers

def read_data(datafile):
    data = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                d = [float(i) for i in line.split()]
                data.append(d)
    return data

def cluster_quality(cluster):
    if len(cluster) == 0:
        return 0.0

    quality = 0.0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            quality += l2_distance(cluster[i], cluster[j])
    return quality / len(cluster)

def compute_quality(data, cluster_indices):
    clusters = dict()
    for i, c in enumerate(cluster_indices):
        if c in clusters:
            clusters[c].append(data[i])
        else:
            clusters[c] = [data[i]]
    return sum(cluster_quality(c) for c in clusters.values())
