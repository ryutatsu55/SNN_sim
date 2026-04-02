import numpy as np

def get_data():
    """テスト用のランダムな時系列データとラベルを生成して返す"""
    # X: (サンプル数, 時間ステップ, 入力次元)
    # Y: (サンプル数, クラス数) ※ワンホットエンコーディング想定
    X_train = np.random.rand(10, 100, 30) 
    Y_train = np.eye(3)[np.random.choice(3, 10)] # 3クラス分類
    X_test = np.random.rand(5, 100, 30)
    Y_test = np.eye(3)[np.random.choice(3, 5)]
    
    return X_train, Y_train, X_test, Y_test