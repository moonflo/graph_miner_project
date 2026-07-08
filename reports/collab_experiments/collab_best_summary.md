# ogbl-collab Best Summary

## Best method by Hits@50 for decay sweep
- method: `weighted_resource_allocation`, Hits@50: `0.597000`, limit_pos: `1000`

## Best decay for time_decay_common_neighbors
- method: `time_decay_common_neighbors`, Hits@50: `0.597000`, limit_pos: `1000`, decay: `0.700000`

## Best decay for time_decay_resource_allocation
- method: `time_decay_resource_allocation`, Hits@50: `0.597000`, limit_pos: `1000`, decay: `0.700000`

## Best method for each positive limit in scale sweep
| limit_pos | best_method | hits_at_50 | decay |
| --- | --- | --- | --- |
| 1000 | adamic_adar | 0.597000 | 0.800000 |
| 5000 | time_decay_common_neighbors | 0.625800 | 0.800000 |
| 10000 | time_decay_common_neighbors | 0.642000 | 0.800000 |
| 20000 | time_decay_common_neighbors | 0.642250 | 0.800000 |

## Improvement of best time-decay method over adamic_adar and resource_allocation
| limit_pos | best_time_decay | time_decay_hits | adamic_adar_hits | delta_vs_adamic_adar | resource_allocation_hits | delta_vs_resource_allocation |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | time_decay_common_neighbors | 0.597000 | 0.597000 | 0.000000 | 0.597000 | 0.000000 |
| 5000 | time_decay_common_neighbors | 0.625800 | 0.610000 | 0.015800 | 0.609800 | 0.016000 |
| 10000 | time_decay_common_neighbors | 0.642000 | 0.628700 | 0.013300 | 0.628300 | 0.013700 |
| 20000 | time_decay_common_neighbors | 0.642250 | 0.629900 | 0.012350 | 0.629600 | 0.012650 |
