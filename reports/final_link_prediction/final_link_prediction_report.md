# Final Link Prediction Experiment Summary

## 1. Basic Setting

- Dataset: `ogbl_collab`
- Split: `valid`
- Train graph: `full train graph`
- Seed: `42`
- Generated at: `2026-07-08T07:46:55.022991+00:00`
- Note: application_candidate results are not directly comparable with the OGB leaderboard.

## 2. Experiments

| Experiment | Eval Mode | Negative Sampling | Tie Policy | Limit Pos | Negatives per Positive | Candidate Size |
|---|---|---|---|---:|---:|---:|
| experiment_1_legacy_baseline | legacy |  |  | 100000 | 100 |  |
| experiment_2_random_candidate_pool | application_candidate | source_fixed_random | average | 1000 | 50 | 51 |
| experiment_3_local_2hop_candidate_pool | application_candidate | source_fixed_2hop | average | 10000 | 50 | 51 |
| experiment_4_full_valid_2hop | application_candidate | source_fixed_2hop | average | 100000 | 50 | 51 |

## 3. Main Results

| Experiment | Method | Hits@1 | Hits@5 | Hits@10 | Hits@20 | Hits@50 | MRR | MeanRank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| experiment_1_legacy_baseline | jaccard | N/A | N/A | N/A | N/A | 0.593769 | N/A | N/A |
| experiment_1_legacy_baseline | adamic_adar | N/A | N/A | N/A | N/A | 0.633729 | N/A | N/A |
| experiment_1_legacy_baseline | resource_allocation | N/A | N/A | N/A | N/A | 0.633530 | N/A | N/A |
| experiment_1_legacy_baseline | preferential_attachment | N/A | N/A | N/A | N/A | 0.107816 | N/A | N/A |
| experiment_1_legacy_baseline | time_decay_common_neighbors | N/A | N/A | N/A | N/A | 0.647394 | N/A | N/A |
| experiment_1_legacy_baseline | time_decay_resource_allocation | N/A | N/A | N/A | N/A | 0.643849 | N/A | N/A |
| experiment_2_random_candidate_pool | jaccard | 0.585000 | 0.597000 | 0.597000 | 0.597000 | 1.000000 | 0.606206 | 11.119500 |
| experiment_2_random_candidate_pool | adamic_adar | 0.588000 | 0.597000 | 0.597000 | 0.597000 | 1.000000 | 0.607456 | 11.118500 |
| experiment_2_random_candidate_pool | resource_allocation | 0.588000 | 0.597000 | 0.597000 | 0.597000 | 1.000000 | 0.607456 | 11.118500 |
| experiment_2_random_candidate_pool | preferential_attachment | 0.117000 | 0.385000 | 0.525000 | 0.704000 | 0.996000 | 0.251027 | 14.859500 |
| experiment_3_local_2hop_candidate_pool | jaccard | 0.310800 | 0.426200 | 0.482500 | 0.533300 | 0.699000 | 0.375993 | 23.079950 |
| experiment_3_local_2hop_candidate_pool | adamic_adar | 0.391400 | 0.487300 | 0.534800 | 0.578100 | 0.701300 | 0.447876 | 21.218150 |
| experiment_3_local_2hop_candidate_pool | resource_allocation | 0.401200 | 0.483600 | 0.527700 | 0.574900 | 0.701300 | 0.452193 | 21.320200 |
| experiment_3_local_2hop_candidate_pool | preferential_attachment | 0.043900 | 0.157200 | 0.263100 | 0.448300 | 0.980500 | 0.126254 | 23.723450 |
| experiment_4_full_valid_2hop | jaccard | 0.320734 | 0.438819 | 0.493143 | 0.544854 | 0.700353 | 0.387435 | 22.591672 |
| experiment_4_full_valid_2hop | adamic_adar | 0.397027 | 0.493093 | 0.534884 | 0.581419 | 0.702417 | 0.452597 | 21.065242 |
| experiment_4_full_valid_2hop | resource_allocation | 0.406880 | 0.489348 | 0.530474 | 0.577891 | 0.702417 | 0.457198 | 21.149674 |
| experiment_4_full_valid_2hop | preferential_attachment | 0.039378 | 0.141668 | 0.241079 | 0.423291 | 0.978414 | 0.117076 | 24.546502 |

## 4. Diagnostics

| Experiment | Method | PosZeroRate | NegZeroRate | AvgTiesWithPos | AvgGreaterThanPos | FallbackRandomNegativeRatio |
|---|---|---:|---:|---:|---:|---:|
| experiment_1_legacy_baseline | jaccard | N/A | N/A | N/A | N/A | N/A |
| experiment_1_legacy_baseline | adamic_adar | N/A | N/A | N/A | N/A | N/A |
| experiment_1_legacy_baseline | resource_allocation | N/A | N/A | N/A | N/A | N/A |
| experiment_1_legacy_baseline | preferential_attachment | N/A | N/A | N/A | N/A | N/A |
| experiment_1_legacy_baseline | time_decay_common_neighbors | N/A | N/A | N/A | N/A | N/A |
| experiment_1_legacy_baseline | time_decay_resource_allocation | N/A | N/A | N/A | N/A | N/A |
| experiment_2_random_candidate_pool | jaccard | 0.403000 | 0.997060 | 20.089000 | 0.075000 | N/A |
| experiment_2_random_candidate_pool | adamic_adar | 0.403000 | 0.997060 | 20.089000 | 0.074000 | N/A |
| experiment_2_random_candidate_pool | resource_allocation | 0.403000 | 0.997060 | 20.089000 | 0.074000 | N/A |
| experiment_2_random_candidate_pool | preferential_attachment | 0.000000 | 0.000000 | 2.333000 | 12.693000 | N/A |
| experiment_3_local_2hop_candidate_pool | jaccard | 0.356000 | 0.065586 | 1.958100 | 21.100900 | 0.065586 |
| experiment_3_local_2hop_candidate_pool | adamic_adar | 0.356000 | 0.065586 | 2.963300 | 18.736500 | 0.065586 |
| experiment_3_local_2hop_candidate_pool | resource_allocation | 0.356000 | 0.065586 | 2.964200 | 18.838100 | 0.065586 |
| experiment_3_local_2hop_candidate_pool | preferential_attachment | 0.000000 | 0.000000 | 1.618300 | 21.914300 | 0.065586 |
| experiment_4_full_valid_2hop | jaccard | 0.351108 | 0.061718 | 1.910725 | 20.636309 | 0.061718 |
| experiment_4_full_valid_2hop | adamic_adar | 0.351108 | 0.061718 | 2.873044 | 18.628720 | 0.061718 |
| experiment_4_full_valid_2hop | resource_allocation | 0.351108 | 0.061718 | 2.873910 | 18.712719 | 0.061718 |
| experiment_4_full_valid_2hop | preferential_attachment | 0.000000 | 0.000000 | 1.635810 | 22.728597 | 0.061718 |

## 5. Short Notes

- The local 2-hop candidate-pool experiment is the main application-oriented setting.
- Hits@50 should be interpreted together with Hits@1/5/10 and MRR.
- Preferential Attachment may obtain high Hits@50 but low MRR, indicating weak top-ranking ability.
- Adamic-Adar and Resource Allocation are the primary recommended interpretable topology methods for system integration.
