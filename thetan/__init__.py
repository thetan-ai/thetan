from thetan.core import sequential, sequential_prelim, reassign_uncertain_update_labels
from thetan.core import centroids_from_labels, candidate_labels_c, assign, reorder_labels
from thetan.berserker import berserker, berserker_dbg, candidate_labels
from thetan.core import sequential_pexact, assign_pexact, assign_hashx
from thetan.metrics import distance_matrix

TS = sequential
TB = berserker
TP = sequential_prelim


