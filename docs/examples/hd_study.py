import numpy as np
import matplotlib.pyplot as plt

def new_metric(X,Y,max=1):
    return np.max(np.abs(X-Y),axis=1)
    # return np.mean(
    #     np.flip(np.sort(np.abs(X-Y),axis=1),axis=1)[:,:max]
    #     ,axis=1
    # )
    #return np.abs(X-Y)


def min_metric(X,Y):
    return np.min(np.abs(X-Y),axis=1)

def thresh_metric(X,Y,thresh = 0.8):
    diff = np.square(X-Y)
    diff[diff<thresh] = 0
    return np.sqrt(np.sum(diff,axis=1))

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def euclidean(p1,p2):
    return np.linalg.norm(p1-p2,axis=1)

dims_list = [2,10,50,100,500,1000]
NUM_RUNS = 5
#dims_list =[2]
distance = "euclidean" # "thresh_metric"
center_type = 'normal'
#SEED = np.random.seed(42)
fig,ax = plt.subplots(3,2,figsize=(10,8))

for i,dims in enumerate(dims_list):
    print(i, dims)
    final_d1_flag = None
    for run in range(NUM_RUNS):
        c1 = np.array([0]*(dims))

        if center_type=='shifted':
            uniform_points = 10*sample_spherical(10000, dims).transpose()
            centre_index = np.random.choice(np.arange(len(uniform_points)))
            c2 = uniform_points[centre_index]
        else:
            c2 = np.array([10]+[0]*(dims-1))

        #SEED = np.random.seed(42)

        norm1 = np.random.multivariate_normal(c1, np.eye(dims), size=50000)
        norm2 = np.random.multivariate_normal(c2, np.eye(dims), size=50000)
        points = np.vstack([norm1,norm2])


        # ## EUCLIDEAN
        if distance=="euclidean":
            d1 = np.linalg.norm(points-c1, axis=1)
        if distance=="new":
            d1 = new_metric(points, c1)
        if distance=="min":
            d1 = min_metric(points, c1)
        if distance=="check_dims":
            displacements = np.abs(points-c1)
        if distance=="thresh_metric":
            THRESH = 6
            d1 = thresh_metric(points, c1, thresh=THRESH)

        if final_d1_flag:
            final_d1 = final_d1+d1
        else:
            final_d1 = d1
            final_d1_flag = 1

    final_d1 = final_d1 / NUM_RUNS
    #diffs = np.square(points-c1)
    ax[i//2][i%2].hist(final_d1,bins=1000)
    # ax[i//2][i%2].hist(final_d1[:50000],bins=1000)
    # ax[i//2][i%2].hist(final_d1[50000:],bins=1000)
    ax[i//2][i%2].set_title(f'Dim: {dims}')

fig.suptitle(f'Euclidean distance histograms: {center_type} centers')
plt.tight_layout()
# plt.savefig(f'/home/elef/tmp/plots/distance_histogram_subplots/euclidean_{center_type}_num_runs_{NUM_RUNS}.png')
plt.show()
