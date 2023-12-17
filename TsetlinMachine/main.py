import struct
import numpy as np
from TsetlinMachine import MultiClassTsetlinMachine
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def float_to_binary(f: float, bits=32) -> np.ndarray:
    packed_data = struct.pack('f', f)

    binary_representation = ''.join(f'{byte:08b}' for byte in bytearray(packed_data))

    trunc = binary_representation[:bits]
    trunc = [ch == '1' for ch in trunc]
    return np.array(trunc, dtype=np.bool_)

def binarize(X: np.ndarray, bits=32) -> np.ndarray:
    result = []
    for x in X:
        bin_data = np.concatenate([float_to_binary(f, bits=bits) for f in x])
        result.append(bin_data)

    return np.stack(result)

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = binarize(X, bits=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mtm = MultiClassTsetlinMachine(input_dim=16, cls_num=3, n=100, s=3.0, T=10)
    mtm.fit(X_train, y_train, epoch=50)

    y_pred_test = mtm.inference(X_test)
    y_pred_train = mtm.inference(X_train)
    
    print(classification_report(y_train, y_pred_train))
    print(classification_report(y_test, y_pred_test))