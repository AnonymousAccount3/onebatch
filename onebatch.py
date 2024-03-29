import gc
import bisect
import numpy as np
from sklearn.metrics import pairwise_distances

class SparseMatrix:
    
    def __init__(self, data, cols=None):
        self.shape = data.shape
        self.cols = []
        self.data = []
        self.rows = (data > 0.)
        self.sum_rows = data.sum(1)

        arange = np.arange(data.shape[0])
        for k in range(data.shape[1]):
            mask = data[:, k] > 0.
            self.cols.append(arange[mask])
            self.data.append(data[:, k][mask])

    def update(self, indices, data):
        
        for k, j in zip(indices, range(len(indices))):
            mask = self.data[k] > data[j]

            self.rows[self.cols[k][~mask], k] = False

            self.sum_rows[self.cols[k][~mask]] -= self.data[k][~mask]
            self.sum_rows[self.cols[k][mask]] -= data[j]

            self.data[k] = self.data[k][mask] - data[j]
            self.cols[k] = self.cols[k][mask]
    
    def get_row(self, j):
        data = []
        cols = np.arange(self.shape[1])[self.rows[j]]
        for k in cols:
            indice = bisect.bisect_left(self.cols[k], j)
            data.append(self.data[k][indice])
        return cols, np.array(data)
   
    
    def sum(self, axis=1):
        if axis==1:
            return self.sum_rows
        elif axis==0:
            return np.array([d.sum() for d in self.data])

        
    def __iter__(self):
        return None
        

    def __del__(self):
        del self.data
        del self.cols
        del self.rows
        del self.sum_rows

    
    def sparse_ratio(self):
        total = np.prod(self.shape)
        non_zeros = sum([len(c) for c in self.cols])
        return non_zeros / total

    
def one_batch_greedy_kmedoids(X, K=1, distance="euclidean", batch_size=1000, verbose=1, sparse_ratio=False):

    indexes = []
    list_sparse_ratios = []
    col_sparse_ratios = []
    med_sparse_ratios = []
    
    batch_size = min(X.shape[0], batch_size)
    
    batch_indexes = np.random.choice(X.shape[0],
                                     batch_size,
                                     replace=False)
    batch_distance = pairwise_distances(X, X[batch_indexes], metric=distance)
    batch_distance *= -1.
    gains = batch_distance.mean(1)
    
    index = gains.argmax()
    indexes.append(index)

    deltas_k = np.zeros(batch_size)
    deltas_k += batch_distance[index]
    
    batch_distance -= deltas_k
    batch_distance.clip(0., None, out=batch_distance)

    batch_distance[batch_indexes, np.arange(batch_size)] *= 0.

    sparse_batch_distance = SparseMatrix(batch_distance)

    deltas = np.copy(-deltas_k)

    del batch_distance
    gc.collect()
    
    for k in range(K-1):
        gains = sparse_batch_distance.sum(axis=1)

        if gains.max() > 1e-4:
            index = gains.argmax()
        else:
            index = np.random.choice(
            list(set(np.arange(X.shape[0])) - set(indexes)))

        indices, data = sparse_batch_distance.get_row(index)

        if sparse_ratio:
            list_sparse_ratios.append(sparse_batch_distance.sparse_ratio())
            col_sparse_ratios.append(sparse_batch_distance.rows[:, indices].mean())
            med_sparse_ratios.append(float(len(indices)/batch_size))
        
        sparse_batch_distance.update(indices, data)

        indexes.append(index)

        if verbose:
           if len(indices) > 0:
               d = np.zeros(batch_size)
               d[indices] += data
               deltas = np.clip(deltas - d, 0., None)

        if verbose:
            obj = np.mean(deltas)
            print("Epoch %i / %i: Estimated Objective %.4f"%(k, K, obj))

    del sparse_batch_distance
    gc.collect()

    if sparse_ratio:
        return dict(index=indexes, ratio=list_sparse_ratios, col_ratio=col_sparse_ratios, med_ratio=med_sparse_ratios)
    else:
        return indexes