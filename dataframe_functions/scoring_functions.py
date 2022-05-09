import numpy as np

def score(df):
    # method="first" breaks ties based on order in array
    pct_ranks = df['prediction'].rank(pct=True, method="first")
    targets = df['target']
    return np.corrcoef(targets, pct_ranks)[0, 1]

def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)