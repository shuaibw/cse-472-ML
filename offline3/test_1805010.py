import pickle
from train_1805010 import Network, DenseLayer, ActivationLayer, DropoutLayer, load_data, extract_data_and_labels
import numpy as np
from sklearn.metrics import f1_score

def load_test(test_data):
    print("Loading test data...")
    X, y = extract_data_and_labels(test_data)
    X = X.T
    y = y - 1
    y_oh = np.eye(26)[y.astype('int32')]
    y_oh = y_oh.T
    return X, y_oh, y

with open('model_1805010.pickle', 'rb') as f:
    model = pickle.load(f)
    
test_data = pickle.load(open('ids7.pickle', 'rb'))
X_test, y_test, y_test_raw = load_test(test_data)

preds, acc = model.predict(X_test, y_test)
print(f"Testing accuracy: {acc:.5f}")
macro_f1 = f1_score(y_test_raw, preds, average='macro')
print(f'Test macro F1 score: {macro_f1:.5f}')

