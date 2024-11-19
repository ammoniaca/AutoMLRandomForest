from enum import Enum


class ClassificationScore(Enum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    TOP_K_ACCURACY = "top_k_accuracy"
    AVERAGE_PRECISION = "average_precision"
    NEG_BRIER_SCORE = "neg_brier_score"
    F1 = "f1"
    F1_MICRO = "f1_micro"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    F1_SAMPLES = "f1_samples"
    NEG_LOG_LOSS = "neg_log_loss"
    PRECISION = "precision"
    RECALL = "recall"
    JACCARD = "jaccard"
    ROC_AUC = "roc_auc"
    ROC_AUC_OVR = "roc_auc_ovr"
    ROC_AUC_OVO = "roc_auc_ovo"
    ROC_AUC_OVR_WEIGHTED = "roc_auc_ovr_weighted"
    ROC_AUC_OVO_WEIGHTED = "roc_auc_ovo_weighted"
    D2_LOG_LOSS_SCORE = "d2_log_loss_score"


class RegressionScore(Enum):
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"
    NEG_MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    NEG_MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    NEG_ROOT_MEAN_SQUARED_ERROR = "neg_root_mean_squared_error"
    NEG_MEAN_SQUARED_LOG_ERROR = "neg_mean_squared_log_error"
    NEG_ROOT_MEAN_SQUARED_LOG_ERROR = "neg_root_mean_squared_log_error"
    NEG_MEDIAN_ABSOLUTE_ERROR = "neg_median_absolute_error"
    R2 = "r2"
    NEG_MEAN_POISSON_DEVIANCE = "neg_mean_poisson_deviance"
    NEG_MEAN_GAMMA_DEVIANCE = "neg_mean_gamma_deviance"
    NEG_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "neg_mean_absolute_percentage_error"
    D2_ABSOLUTE_ERROR_SCORE = "d2_absolute_error_score"