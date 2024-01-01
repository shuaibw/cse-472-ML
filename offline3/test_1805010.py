import pickle
from train_1805010 import Network, DenseLayer, ActivationLayer, DropoutLayer, load_data
from sklearn.metrics import f1_score

with open('model_1805010.pickle', 'rb') as f:
    model = pickle.load(f)

X_test, y_test, y_test_raw = load_data(train=False)
preds, acc = model.predict(X_test, y_test)
print(f"Testing accuracy: {acc:.5f}")
macro_f1 = f1_score(y_test_raw, preds, average='macro')
print(f'Test macro F1 score: {macro_f1:.5f}')

