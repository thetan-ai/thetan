from time import time

from dipy.io.image import load_nifti
from dipy.viz import skyline

from thetan.berserker import cluster_merge

# path = r"/Users/prajwal/Desktop/t1_data/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz"
# path = r'/home/prajwal/Downloads/IXI002-Guys-0828-T1.nii.gz'
# path = r'/home/prajwal/Downloads/t1_data/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz'
# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
fname = '/home/elef/Data/T1_from_Jasmine/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz'

data, affine = load_nifti(fname)

# Parameters
theta = 50
difference = 30
max_difference = 2 * 40

start_time = time()
images = cluster_merge(data, affine, theta=theta, threshold=difference, max_intensity_threshold=max_difference, individual_labels=True, verbose=True)
print(f"Clobbering took {time() - start_time:.2f} seconds")

skyline(images = images)