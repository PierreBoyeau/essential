"""TF → gene-expression NB regression.

Public surface:
    prepare_layers, TFStandardizer, TFArrays          — data
    build_tf_mask                                     — RegulonDB → mask
    load_synonym_rows, reconcile_names                — RegulonDB name reconciliation
    MetabolicModel, build_gt_labels, GTLabels         — iML1515 → GT labels
    TFRegression, ConstantMean, TFCellBox             — models
    TFRegressionFixed                                 — TFRegression w/ log link
    TFRegressionNuisance, TFCellBoxNuisance           — models w/ nuisance covariates
    lcp_to_count_mean, nb_nll                         — likelihood building blocks
    fit                                               — training loop
    per_gene_nll, per_perturbation_nll                — NLL evaluators
    per_perturbation_moments, tf_zscores              — NB-residual diagnostics
    mask_perturbed_gene                               — drop perturbed gene from moments
    tf_score_frame                                    — z-scores → long DataFrame
    pr_curve                                          — PR curve vs GT labels
    PerPertNLL, PerPertMoments, TFZScores, PRCurve    — eval return types
    profile_metrics                                   — LFC / DEG metrics
"""

from .data import TFArrays, TFStandardizer, prepare_layers
from .eval import (
    PerPertMoments,
    PerPertNLL,
    PRCurve,
    TFZScores,
    mask_perturbed_gene,
    per_gene_nll,
    per_perturbation_moments,
    per_perturbation_nll,
    pr_curve,
    tf_score_frame,
    tf_zscores,
)
from .metabolic import GTLabels, MetabolicModel, build_gt_labels
from .metrics import profile_metrics
from .models import (
    ConstantMean,
    TFCellBox,
    TFCellBoxNuisance,
    TFRegression,
    TFRegressionFixed,
    TFRegressionNuisance,
    lcp_to_count_mean,
    nb_nll,
)
from .regulondb import build_tf_mask, load_synonym_rows, reconcile_names
from .train import fit

__all__ = [
    "ConstantMean",
    "GTLabels",
    "MetabolicModel",
    "PRCurve",
    "PerPertMoments",
    "PerPertNLL",
    "TFArrays",
    "TFCellBox",
    "TFCellBoxNuisance",
    "TFRegression",
    "TFRegressionFixed",
    "TFRegressionNuisance",
    "TFStandardizer",
    "TFZScores",
    "build_gt_labels",
    "build_tf_mask",
    "fit",
    "lcp_to_count_mean",
    "load_synonym_rows",
    "mask_perturbed_gene",
    "nb_nll",
    "per_gene_nll",
    "per_perturbation_moments",
    "per_perturbation_nll",
    "pr_curve",
    "prepare_layers",
    "profile_metrics",
    "reconcile_names",
    "tf_score_frame",
    "tf_zscores",
]
