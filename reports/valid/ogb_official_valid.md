# ogbl-collab OGB Official Evaluation

Evaluation mode: OGB official-style via `ogb.linkproppred.Evaluator`.
Use this report for leaderboard-aligned comparisons before legacy smoke results.

Sanity check: official Adamic-Adar for ogbl-collab is expected near `0.6349` Hits@50 on the valid split. Large deviations usually mean the negative layout, train graph boundary, valid/test visibility, or undirected projection needs inspection.

| dataset | split | method | decay | pos_used | neg_per_pos_used | total_neg_used | hits@50 | nodes | edges | has_weight | has_year | max_train_year | runtime_seconds | official_mode | official_negatives_available | edge_neg_shape | negative_layout | y_pred_pos_shape | y_pred_neg_shape | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbl_collab | valid | adamic_adar | 0.800000 | 60084 | N/A | 100 | 0.648892 | 235868 | 967632 | yes | yes | 2017 | 0.801 | yes | yes | 100x2 | shared_pool | 60084 | 100 |  |
| ogbl_collab | valid | resource_allocation | 0.800000 | 60084 | N/A | 100 | 0.648892 | 235868 | 967632 | yes | yes | 2017 | 0.745 | yes | yes | 100x2 | shared_pool | 60084 | 100 |  |
| ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 60084 | N/A | 100 | 0.648892 | 235868 | 967632 | yes | yes | 2017 | 2.992 | yes | yes | 100x2 | shared_pool | 60084 | 100 |  |
| ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 60084 | N/A | 100 | 0.648892 | 235868 | 967632 | yes | yes | 2017 | 135.652 | yes | yes | 100x2 | shared_pool | 60084 | 100 |  |

If `negative_layout` is `shared_pool`, the split exposed 2D `edge_neg` and `y_pred_neg` was passed to the installed OGB Hits evaluator as a 1D shared negative pool, not as per-positive row-wise negatives.
