# ogbl-collab Classical Link Prediction Experiments

| experiment_group | dataset | split | method | decay | limit_pos | positive_split_full | pos_used | requested_neg | available_neg | neg_used | negative_truncated | hits_at_50 | num_nodes | num_edges | has_weight | has_year | max_train_year | topology_edges | runtime_seconds | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.700000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 19.083 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 0.700000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 19.083 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.700000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 19.083 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.700000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 19.083 |  |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.912 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.912 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.912 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.912 |  |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.850000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.950 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 0.850000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.950 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.850000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.950 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.850000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 15.950 |  |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.900000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.272 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 0.900000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.272 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.900000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.272 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.900000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.272 |  |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.950000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.120 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 0.950000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.120 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.950000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.120 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.950000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.120 |  |
| decay_sweep | ogbl_collab | valid | weighted_resource_allocation | 1.000000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.220 |  |
| decay_sweep | ogbl_collab | valid | weighted_adamic_adar | 1.000000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.220 |  |
| decay_sweep | ogbl_collab | valid | time_decay_common_neighbors | 1.000000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.220 |  |
| decay_sweep | ogbl_collab | valid | time_decay_resource_allocation | 1.000000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.220 |  |
| scale_sweep | ogbl_collab | valid | adamic_adar | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.210 |  |
| scale_sweep | ogbl_collab | valid | resource_allocation | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.210 |  |
| scale_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.210 |  |
| scale_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.210 |  |
| scale_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 1000 | no | 1000 | 50000 | 100000 | 50000 | no | 0.597000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 16.210 |  |
| scale_sweep | ogbl_collab | valid | adamic_adar | 0.800000 | 5000 | no | 5000 | 250000 | 100000 | 100000 | yes | 0.610000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 34.424 |  |
| scale_sweep | ogbl_collab | valid | resource_allocation | 0.800000 | 5000 | no | 5000 | 250000 | 100000 | 100000 | yes | 0.609800 | 235868 | 967632 | yes | yes | 2017 | 967632 | 34.424 |  |
| scale_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.800000 | 5000 | no | 5000 | 250000 | 100000 | 100000 | yes | 0.610400 | 235868 | 967632 | yes | yes | 2017 | 967632 | 34.424 |  |
| scale_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 5000 | no | 5000 | 250000 | 100000 | 100000 | yes | 0.625800 | 235868 | 967632 | yes | yes | 2017 | 967632 | 34.424 |  |
| scale_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 5000 | no | 5000 | 250000 | 100000 | 100000 | yes | 0.622200 | 235868 | 967632 | yes | yes | 2017 | 967632 | 34.424 |  |
| scale_sweep | ogbl_collab | valid | adamic_adar | 0.800000 | 10000 | no | 10000 | 500000 | 100000 | 100000 | yes | 0.628700 | 235868 | 967632 | yes | yes | 2017 | 967632 | 49.859 |  |
| scale_sweep | ogbl_collab | valid | resource_allocation | 0.800000 | 10000 | no | 10000 | 500000 | 100000 | 100000 | yes | 0.628300 | 235868 | 967632 | yes | yes | 2017 | 967632 | 49.859 |  |
| scale_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.800000 | 10000 | no | 10000 | 500000 | 100000 | 100000 | yes | 0.628500 | 235868 | 967632 | yes | yes | 2017 | 967632 | 49.859 |  |
| scale_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 10000 | no | 10000 | 500000 | 100000 | 100000 | yes | 0.642000 | 235868 | 967632 | yes | yes | 2017 | 967632 | 49.859 |  |
| scale_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 10000 | no | 10000 | 500000 | 100000 | 100000 | yes | 0.640200 | 235868 | 967632 | yes | yes | 2017 | 967632 | 49.859 |  |
| scale_sweep | ogbl_collab | valid | adamic_adar | 0.800000 | 20000 | no | 20000 | 1000000 | 100000 | 100000 | yes | 0.629900 | 235868 | 967632 | yes | yes | 2017 | 967632 | 84.245 |  |
| scale_sweep | ogbl_collab | valid | resource_allocation | 0.800000 | 20000 | no | 20000 | 1000000 | 100000 | 100000 | yes | 0.629600 | 235868 | 967632 | yes | yes | 2017 | 967632 | 84.245 |  |
| scale_sweep | ogbl_collab | valid | weighted_resource_allocation | 0.800000 | 20000 | no | 20000 | 1000000 | 100000 | 100000 | yes | 0.630650 | 235868 | 967632 | yes | yes | 2017 | 967632 | 84.245 |  |
| scale_sweep | ogbl_collab | valid | time_decay_common_neighbors | 0.800000 | 20000 | no | 20000 | 1000000 | 100000 | 100000 | yes | 0.642250 | 235868 | 967632 | yes | yes | 2017 | 967632 | 84.245 |  |
| scale_sweep | ogbl_collab | valid | time_decay_resource_allocation | 0.800000 | 20000 | no | 20000 | 1000000 | 100000 | 100000 | yes | 0.641900 | 235868 | 967632 | yes | yes | 2017 | 967632 | 84.245 |  |
