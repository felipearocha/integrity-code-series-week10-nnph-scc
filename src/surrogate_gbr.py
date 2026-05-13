"""GBR surrogate for Week 10 - 8-parameter inputs."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from src.monte_carlo import PARAM_NAMES

def build_X(params):
    return np.column_stack([params[k] for k in PARAM_NAMES])

def train_surrogate(params, wl, n_estimators=300, max_depth=5, lr=0.05,
                     subsample=0.8, seed=42, test_size=0.15):
    X=build_X(params); y=wl
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    gbr=GradientBoostingRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                   learning_rate=lr,subsample=subsample,
                                   random_state=seed,loss="huber")
    gbr.fit(Xtr,ytr); yp_tr=gbr.predict(Xtr); yp_te=gbr.predict(Xte)
    perm=permutation_importance(gbr,Xte,yte,n_repeats=10,random_state=seed,scoring="r2")
    return {"model":gbr,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,
            "y_pred_train":yp_tr,"y_pred_test":yp_te,
            "r2_train":r2_score(ytr,yp_tr),"r2_test":r2_score(yte,yp_te),
            "mae_test":mean_absolute_error(yte,yp_te),
            "feature_importance":perm.importances_mean,"feature_names":PARAM_NAMES}
