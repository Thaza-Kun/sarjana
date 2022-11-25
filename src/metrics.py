from sklearn.metrics import fbeta_score, make_scorer

f2_score = make_scorer(fbeta_score, beta=2)

f_half_score = make_scorer(fbeta_score, beta=0.5)
