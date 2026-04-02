import numpy as np

def train_ridge(X_feat, Y_target, lambda_reg=1.0):
    """Ridge回帰で重み(W_out)を計算するダミー処理"""
    print(" [Model] Training Ridge Regression...")
    # 特徴量次元 x ターゲット次元 の重み行列を返す
    feat_dim = X_feat.shape[-1]
    target_dim = Y_target.shape[-1]
    return np.random.rand(feat_dim, target_dim)

def predict_ridge(X_feat, w_out):
    """計算済みの重みを使って予測を行う"""
    # 実際の内積と積分処理などの代わり
    samples = X_feat.shape[0]
    target_dim = w_out.shape[1]
    return np.random.rand(samples, target_dim)