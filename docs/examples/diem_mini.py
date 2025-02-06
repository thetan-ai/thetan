import numpy as np
from scipy.spatial.distance import cdist

def diem_stat(N, maxV, minV, num_samples=int(1e5)):
    # Generate uniform distributions
    a = np.random.uniform(minV, maxV, (num_samples, N, 1))
    b = np.random.uniform(minV, maxV, (num_samples, N, 1))

    # Compute Euclidean distances
    d = np.linalg.norm(a - b, axis=1)

    # Compute DIEM statistics
    exp_center = np.median(d)
    vard = np.var(d)  # Variance of Euclidean distance distribution
    adjusted_dist = (maxV - minV) * (d - exp_center) / vard  # Centering and scaling DIEM
    std_one = np.std(adjusted_dist)  # One standard deviation of DIEM

    min_DIEM = -(maxV - minV) * (exp_center / vard)  # Minimum DIEM
    max_DIEM = (maxV - minV) * (np.sqrt(N) * (maxV - minV) - exp_center) / vard  # Maximum DIEM

    return exp_center, vard, std_one, min_DIEM, max_DIEM

def diem(synMat1, synMat2, maxV, minV, exp_center, vard):
    # Compute DIEM similarities
    DIEM = (maxV - minV) * ((cdist(synMat1.T, synMat2.T, metric='euclidean') - exp_center) / vard)

    # Ensure symmetric and upper triangular if needed
    if np.allclose(DIEM, DIEM.T):
        DIEM = np.triu(DIEM)
        DIEM[DIEM == 0] = np.nan

    return DIEM

if __name__ == '__main__':
    N = 100000

    for D in range(2, 200, 2):
        print(f"\nDimension: {D}")

        r1 = np.random.rand(D, N)
        center = r1.mean(axis=1, keepdims=True)
        res = cdist(r1.T, center.T)

        corner = np.full((1, D), D)
        res2 = cdist(corner, center.T)
        print(f"D: {D}, min: {res.min()}, max: {res.max()}, corner_center: {res2[0,0]}")

        for shift in range(11):
            print(f"\n>>> Shift: {shift}")

            synMat1 = r1 + shift
            synMat2 = r1

            maxV, minV = synMat1.max(), synMat1.min()
            print(f"minV: {minV}, maxV: {maxV}")

            # Compute DIEM statistics
            exp_center, vard, std_one, min_DIEM, max_DIEM = diem_stat(D, maxV, minV)
            print(f"exp_center: {exp_center}, vard: {vard}, std_one: {std_one}, min_DIEM: {min_DIEM}, max_DIEM: {max_DIEM}")

            # Compute and print DIEM results
            res = diem(synMat1, synMat2, maxV, minV, exp_center, vard)
            print(f"L2D Res >> min: {np.nanmin(res)}, mean: {np.nanmean(res)}, max: {np.nanmax(res)}, std: {np.nanstd(res)}")

            res2 = cdist(synMat1.T, synMat2.T, metric='euclidean')
            print(f"L2 Res >> min: {res2.min()}, mean: {res2.mean()}, max: {res2.max()}, std: {res2.std()}")
