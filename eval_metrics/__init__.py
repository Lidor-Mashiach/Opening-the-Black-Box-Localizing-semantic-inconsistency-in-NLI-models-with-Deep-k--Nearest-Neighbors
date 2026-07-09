"""eval_metrics - all evaluation metrics used by Phase A and Phase B.

One file per metric, exactly as defined in the methodology document:

Output-consistency metrics
    relaxed_fooling_rate      (anchor paper, Arakelyan et al. 2024)
    strict_fooling_rate       (anchor paper, Arakelyan et al. 2024)
    paraphrastic_consistency  (Srikanth et al. 2024, "How Often Are Errors...")
    pattern_accuracy          (Zgreaban et al. 2025, MERGE)

Internal-representation metrics
    layer_distance            (per-layer cosine geometry)
    dknn_credibility          (Papernot & McDaniel 2018, Deep k-NN)
"""
from .relaxed_fooling_rate import relaxed_fooling_rate
from .strict_fooling_rate import strict_fooling_rate
from .paraphrastic_consistency import paraphrastic_consistency
from .pattern_accuracy import pattern_accuracy
from .layer_distance import (
    pairwise_layer_distances,
    mean_distance_profile,
    distance_threshold_per_layer,
    first_crossing_layer,
    layer_region,
)
from .dknn_credibility import (
    DkNN,
    credibility_from_neighbor_labels,
    credibility_drop,
    anomalous_drop_threshold,
    select_best_k,
    diagnose_credibility_scenario,
)
