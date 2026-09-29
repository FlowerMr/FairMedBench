# Fairness methodology

Fairness in medical imaging is context-dependent. A lower numerical gap is not automatically evidence of a clinically fairer system, because the relevant error trade-offs, disease prevalence, sampling process, subgroup size, and clinical use case matter.

FairMedBench therefore avoids a composite fairness score. It reports subgroup-specific metrics and transparent gaps.

## Equal opportunity gap

For a binary task, equal opportunity is related to equality of true-positive rates:

TPR_g = TP_g / (TP_g + FN_g)

The reported gap is:

max_g(TPR_g) - min_g(TPR_g)

## Demographic parity gap

The reported descriptive gap is the range of positive prediction rates:

max_g(P(pred=1 | g)) - min_g(P(pred=1 | g))

This metric may be inappropriate for some medical decision settings because disease prevalence can legitimately differ between populations.

## Interpretation

Subgroup results should be accompanied by subgroup sample sizes and confidence intervals in a research-grade extension. Small groups can produce unstable estimates. Metadata errors can also create misleading disparities.
