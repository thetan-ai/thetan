import numpy as np
import numpy.testing as npt
from thetan.metrics import distance_matrix

def euclidean_ground(x,y):
    return np.linalg.norm(x-y)

def ssd_ground(x,y):
    return np.sum((x-y)**2)

def norm_ssd_ground(x,y):
    return np.mean((x-y)**2)

def dot_ground(x,y):
    return np.dot(x,y)

def l1_ground(x,y):
    return np.sum(np.abs(x-y))

def cosine_ground(x,y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 1.0
    sim = np.dot(x,y) / (norm_x * norm_y)
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim

def arccos_ground(x,y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    sim = np.dot(x,y) / (norm_x * norm_y)
    sim = np.clip(sim, -1.0, 1.0)
    return np.arccos(sim)

ground_metrics = {
    'euclidean': euclidean_ground,
    'ssd':ssd_ground,
    'norm_ssd': norm_ssd_ground,
    'norm_ssd_un': norm_ssd_ground,
    'dot_prod' : dot_ground,
    'l1_norm' : l1_ground,
    'cosine' : cosine_ground,
    'arccos': arccos_ground
}

def ground_distance_matrix(X1, X2, metric_name):
    x,y = X1.shape[0], X2.shape[0]
    mat = np.zeros((x,y))
    func = ground_metrics[metric_name]
    for i in range(x):
        for j in range(y):
            mat[i,j] = func(X1[i], X2[j])
    return mat

def test_distance_matrix():
    rng = np.random.default_rng(42)
    X1 = rng.random((5,10)).astype(np.float64)
    X2 = rng.random((5,10)).astype(np.float64)
    
    metrics = ground_metrics.keys()

    for metric in metrics:
        print('Testing metric: ', metric)
        expected = ground_distance_matrix(X1, X2, metric)
        actual = distance_matrix(X1, X2, metric)

        npt.assert_allclose(actual, expected)

if __name__ == '__main__':
    test_distance_matrix()
    print('Tests passed!')