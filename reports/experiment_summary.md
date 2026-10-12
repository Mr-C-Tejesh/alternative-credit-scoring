# Experiment Summary

**Experiment ID:** SYN-BASELINE-001  
**Status:** Completed as an engineering demo only  
**Dataset:** 10,000 generated synthetic rows  
**Seed:** 2026  
**Split:** Stratified random 80/20 train/test  
**Target:** `default_within_12_months` (synthetic label)

## Held-out test metrics

See `test_metrics.csv` and `metric_comparison.png` for the generated results. Metrics refer exclusively to the generated test split.

| Model | ROC-AUC | Average precision | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Traditional baseline | 0.8071 | 0.6099 | 0.1519 | 0.4611 |
| Hybrid Logistic Regression | 0.9098 | 0.8126 | 0.1034 | 0.3298 |
| Hybrid HistGradientBoosting | 0.9063 | 0.7968 | 0.1060 | 0.3386 |

## Interpretation limit

The alternative-feature signal was explicitly encoded in the data generator. These results confirm that the pipeline executes and can recover patterns in the constructed scenario. They do not demonstrate the predictive value of utility, rental or digital-payment features in real populations, nor do they establish novelty, fairness or business benefit.
