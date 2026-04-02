import numpy as np

def calc_accuracy(y_true, y_pred):
    """正解ラベルと予測結果から精度を計算する"""
    print(" [Eval] Calculating Accuracy...")
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    acc = np.mean(pred_labels == true_labels)
    return acc