# Offline evaluation report

Generated: 2026-07-19

**Status: Engineering smoke test — not evidence of recommendation quality.**

- Only 9 playlists are labeled; at least 50 are required before making recommendation-quality claims.

## Data and policy

- Item catalog: `catalog-v1-078c63e6d23c1d41.parquet` (2,206,451 unique rows).
- Popularity-eligible catalog: 474,725 tracks; bounded candidate sample: 100,000.
- Membership labels: 773 rows across 9 playlists.
- Repeated splits: 20 per playlist, 5 matched seeds per split.
- Intervals: 95% playlist-clustered bootstrap, 2,000 resamples after averaging splits within each playlist.
- The `deployed` row uses the same weighted cosine, PCA(5), minimum-popularity filter, bounded catalog sample, and weighted top-pool randomization as the first web-app request. A fixed random seed is supplied only for reproducibility.
- Session history is not simulated; this is a first-request playlist-continuation benchmark.
- Source matching: 2,036/3,920 tracks (51.94%).

## Strategy comparison

| Strategy | Recall@10 | 95% CI | NDCG@10 | 95% CI | Hit rate | Retrieval ceiling | Retrievable recall | Steering target distance | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_cosine_pca | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.003] | 1.11% | 8.80% | 0.001 | 0.214 | 0.3503% |
| weighted_cosine | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.002] | 1.67% | 8.80% | 0.002 | — | 0.3421% |
| unweighted_cosine | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.002] | 1.11% | 8.80% | 0.001 | — | 0.3387% |
| deployed | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.002] | 0.56% | 8.80% | 0.001 | — | 0.3577% |
| tuned_deployed | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.002] | 0.56% | 8.80% | 0.001 | — | 0.3600% |
| tuned_weighted_cosine_pca | 0.000 | [0.000, 0.000] | 0.001 | [0.000, 0.002] | 0.56% | 8.80% | 0.001 | — | 0.3545% |
| weighted_cosine_pca_steered | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.001] | 0.56% | 8.80% | 0.001 | 0.077 | 0.3324% |
| popularity | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.00% | 8.80% | 0.000 | — | 0.0021% |
| random | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.00% | 8.80% | 0.000 | — | 0.3760% |

Recall uses every non-seed playlist item as relevant. `Retrieval ceiling` is the fraction of those positives present in the actual filtered, bounded candidate pool. `Retrievable recall` conditions on that pool to separate ranking from retrieval loss. Catalog coverage is computed once per strategy as distinct recommended items divided by the full popularity-eligible catalog, so it can distinguish strategies across repeated requests.

## Ablations

| Strategy | Weights | PCA | Randomized | Steering |
|---|---:|---:|---:|---|
| random | no | no | no | off |
| popularity | no | no | no | off |
| unweighted_cosine | no | no | no | off |
| weighted_cosine | yes | no | no | off |
| weighted_cosine_pca | yes | yes | no | off |
| deployed | yes | yes | yes | off |
| weighted_cosine_pca_steered | yes | yes | no | energy=+0.25, valence=+0.25 |
| tuned_weighted_cosine_pca | yes | yes | no | off |
| tuned_deployed | yes | yes | yes | off |

The fixed steering row is not treated as a user-preference label. It is paired with the unsteered PCA row to measure target-distance movement and any relevance cost; playlist NDCG alone cannot validate steering quality.

## Interpretation

This report validates the evaluation machinery only. Collect at least 50 diverse, accessible playlists, rebuild the membership labels so unmatched tracks are retained, and rerun before placing quality numbers on a resume.

The benchmark is a playlist-continuation proxy rather than a direct measure of user satisfaction. Weight selection must occur on a playlist-level tuning partition, with the reported test playlists kept untouched.
