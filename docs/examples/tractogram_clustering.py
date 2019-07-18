import numpy as np
from nibabel import trackvis as tv
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.viz import window, actor
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundlesX

import time
from fury import window, actor
def viz(actor):
    
    ren = window.Renderer()
    ren.add(actor)
    window.show(ren)
    
tractogram, hr = load_trk("whole_brain_MNI.trk")

start_time = time.time()

streamliness = set_number_of_points(tractogram, nb_points=20)

metric = AveragePointwiseEuclideanMetric()
rng = np.random.RandomState(42)
ordering = np.arange(len(streamliness))
merged_clusters = Streamlines()

qbx = QuickBundlesX(thresholds=[40., 25., 15], metric=metric)

start_time = time.time()
clusters_of_layer3_all_qbx = []

for i in range(10):
    
    rng.shuffle(ordering)
    clusters = qbx.cluster(streamliness, ordering=ordering)

    clusters_of_layer3_all_qbx.append(clusters.get_clusters(3))
    

    save_trk('clusters_'+str(i)+'_layer3.trk', clusters.get_clusters(3).centroids, affine=np.eye(4))
    
    merged_clusters.extend(clusters.get_clusters(3).centroids) # last layer centroids
 
    
print("--- %s seconds ---" % (time.time() - start_time))


#merged clusters get clustered

start_time = time.time()

streamlines_m = set_number_of_points(merged_clusters, nb_points=20)


rng = np.random.RandomState(42)

clusters_m = qbx.cluster(streamlines_m)

print("--- %s seconds ---" % (time.time() - start_time))

save_trk('clusters_final_layer3.trk', clusters_m.get_clusters(3).centroids, affine=np.eye(4))

cs = clusters_m.get_clusters(3) # layer
all_c = cs.clusters
    
cen_size_all = []
count = 0
for i in range(10):
    
    count += len(clusters_of_layer3_all_qbx[i].centroids)
    cen_size_all.append(count)
    count += 1
    
all_c = cs.get_large_clusters(5)

for j in range(len(all_c)):

    ct = all_c[j] # cluster number
    cb = Streamlines()

    ci = 0

    for cl in range(len(ct)):

        for i in range(len(cen_size_all)):

            if cen_size_all[i] - ct[cl] > 0:
                ci = i
                break

        if ci == 0 :
            cp = 0
        else:
            cp = ct[cl] - cen_size_all[ci-1]
            cp -= 1

        d = clusters_of_layer3_all_qbx[ci].clusters[cp].indices
        b = tractogram[d]

        cb.extend(b)
        
    #save_trk("final_c"+str(j)+".trk", cb, affine=np.eye(4))

    
save_trk("final_all_c.trk", cb, affine=np.eye(4))

# assign streamlines to merged centroids

from dipy.segment.metric import mdf

all_centroids = clusters_m.get_clusters(3).centroids

ct_st = []
for i in range(len(all_centroids)):
    
    ct_st.append([])
    
for st in range(len(tractogram)):
    
    mdf_min = 1000000000000000
    ct_number = 0
    
    for ct in range(len(all_centroids)):
        
        mdf_val = mdf(tractogram[st], all_centroids[ct])
        
        if mdf_min > mdf_val:
            
            mdf_min = mdf_val
            ct_number = ct
    ct_st[ct_number].append(st)
       
# if you wish to save the clusters with streamlines individually .
for k in range(len(all_centroids)):
    
    if len(ct_st[k]) > 5 : 
        
        save_trk("final_atlas_c"+str(k)+".trk", tractogram[ct_st[k]], affine=np.eye(4))
        
# save clusters from any one of the runs of qbx 
# to compare with final clusters
css = clusters_of_layer3_all_qbx[1].clusters 
cf = Streamlines()
for j in range(len(css)):
    cf = tractogram[css[j].indices]
    if len(cf) > 5 :
        save_trk("initial_atlas_c"+str(j)+".trk", cf, affine=np.eye(4))
