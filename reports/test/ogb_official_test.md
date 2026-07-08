# ogbl-collab OGB Official Evaluation

Evaluation mode: OGB official-style via `ogb.linkproppred.Evaluator`.
Use this report for leaderboard-aligned comparisons before legacy smoke results.

Sanity check: official Adamic-Adar for ogbl-collab is expected near `0.6417` Hits@50 on the test split. Large deviations usually mean the negative layout, train graph boundary, valid/test visibility, or undirected projection needs inspection.

| dataset | split | method | decay | pos_used | neg_per_pos_used | total_neg_used | hits@50 | nodes | edges | has_weight | has_year | max_train_year | runtime_seconds | official_mode | official_negatives_available | edge_neg_shape | negative_layout | y_pred_pos_shape | y_pred_neg_shape | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbl_collab | test | adamic_adar | 0.800000 | 46329 | N/A | 100 | 0.545404 | 235868 | 967632 | yes | yes | 2017 | 0.540 | yes | yes | 100x2 | shared_pool | 46329 | 100 |  |
| ogbl_collab | test | resource_allocation | 0.800000 | 46329 | N/A | 100 | 0.545404 | 235868 | 967632 | yes | yes | 2017 | 0.486 | yes | yes | 100x2 | shared_pool | 46329 | 100 |  |
| ogbl_collab | test | time_decay_common_neighbors | 0.800000 | 46329 | N/A | 100 | 0.545404 | 235868 | 967632 | yes | yes | 2017 | 1.451 | yes | yes | 100x2 | shared_pool | 46329 | 100 |  |
| ogbl_collab | test | time_decay_resource_allocation | 0.800000 | 46329 | N/A | 100 | 0.545404 | 235868 | 967632 | yes | yes | 2017 | 41.847 | yes | yes | 100x2 | shared_pool | 46329 | 100 |  |

If `negative_layout` is `shared_pool`, the split exposed 2D `edge_neg` and `y_pred_neg` was passed to the installed OGB Hits evaluator as a 1D shared negative pool, not as per-positive row-wise negatives.
