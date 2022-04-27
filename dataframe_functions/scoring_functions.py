import scipy
import numpy

def correlation_score_cv(y_true, y_pred):
    return stats.spearmanr(y_true, y_pred)[0]

def correlation_score(y_true, y_pred):
    return 'correlation', stats.spearmanr(y_true, y_pred)[0], True

def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby(era).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

def score(df):
    # method="first" breaks ties based on order in array
    pct_ranks = df[PREDICTION_NAME].rank(pct=True, method="first")
    targets = df[TARGET_NAME]
    return np.corrcoef(targets, pct_ranks)[0, 1]

def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)