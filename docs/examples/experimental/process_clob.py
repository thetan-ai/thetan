from dipy.io.image import load_nifti
from dipy.viz import skyline

from thetan.berserker import cluster_merge

from time import time

path = r"/Users/prajwal/Desktop/t1_data/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz"
data, affine = load_nifti(path)

# Parameters
theta = 50
threshold = 20
max_intensity_threshold = 40

start_time = time()
images = cluster_merge(data, affine, theta=theta, threshold=threshold, max_intensity_threshold=max_intensity_threshold)
print(f"Clobbering took {time() - start_time:.2f} seconds")

skyline(images = images)