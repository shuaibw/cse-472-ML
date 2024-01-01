import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
import gc
import torchvision.datasets as ds
import torchvision.transforms as transforms
import pickle

class Layer:
    def __init__(self):
        pass

    def forward(self, A):
        pass

    def backward(self, dZ):
        pass

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2/input_size) # He initialization
        self.b = np.zeros((output_size, 1))
        self.activation_function = ActivationLayer(activation)

    def forward(self, A):
        Z = self.W.dot(A) + self.b
        assert Z.shape == (self.W.shape[0], A.shape[1])
        return Z

    def backward(self, dZ, A_prev):
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        
        assert dA_prev.shape == A_prev.shape
        assert dW.shape == self.W.shape
        assert db.shape == self.b.shape
        
        return dA_prev, dW, db

class DropoutLayer(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, A):
        self.mask = np.random.rand(*A.shape) > self.rate
        A = A * self.mask
        A /= (1 - self.rate)
        return A

    def backward(self, dA):
        dA = dA * self.mask
        dA /= (1 - self.rate)
        return dA

class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation
        self.activation_cache = None

    def forward(self, Z):
        self.activation_cache = Z
        if self.activation == "relu":
            A = np.maximum(0, Z)
            assert A.shape == Z.shape
            return A
        elif self.activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
            assert A.shape == Z.shape
            return A
        elif self.activation == "softmax":
            Z = np.clip(Z, -20, 20)
            e_x = np.exp(Z)
            A = e_x / np.sum(e_x, axis=0)
            assert A.shape == Z.shape
            return A + 1e-8

    def backward(self, dA):
        Z = self.activation_cache
        if self.activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            assert dZ.shape == Z.shape
            return dZ
        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)
            assert dZ.shape == Z.shape
            return dZ
        elif self.activation == "softmax":
            # For this case, dA = Y_true
            Z = np.clip(Z, -20, 20)
            e_x = np.exp(Z)
            s = e_x / np.sum(e_x, axis=0)
            dZ = s - dA
            assert dZ.shape == Z.shape
            return dZ
class Network:
    def __init__(self):
        self.layers = []
        self.cache = []
        self.costs = []
        self.train_acc = []
        self.test_acc = []
        self.test_costs = []
        # For adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X, training=True):
        A = X
        self.cache = [(None, X)]
        for layer in self.layers:
            if isinstance(layer, DropoutLayer) and training:
                A = layer.forward(A)
            elif isinstance(layer, DenseLayer):
                Z = layer.forward(A)
                A = layer.activation_function.forward(Z)
                self.cache.append((Z, A))
        assert A.shape == (26, X.shape[1])
        return A + 1e-8

    def backward(self, Y, AL):
        gradients = {}
        Y = Y.reshape(AL.shape)
        L = len(self.layers)
        dA_prev = Y  # Assuming last layer uses softmax
        self.cache.pop()
        for i in reversed(range(L)):
            layer = self.layers[i]
            if isinstance(layer, DropoutLayer):
                dA_prev = layer.backward(dA_prev)
            elif isinstance(layer, DenseLayer):
                self.t += 1
                Z, A_prev = self.cache.pop()
                dZ = layer.activation_function.backward(dA_prev)
                dA_prev, dW, db = layer.backward(dZ, A_prev)
                if self.optimizer == "adam":
                    gradients["vdW" + str(i+1)] = self.beta1 * gradients.get("vdW" + str(i+1), np.zeros_like(dW)) + (1 - self.beta1) * dW
                    gradients["vdb" + str(i+1)] = self.beta1 * gradients.get("vdb" + str(i+1), np.zeros_like(db)) + (1 - self.beta1) * db
                    gradients["sdW" + str(i+1)] = self.beta2 * gradients.get("sdW" + str(i+1), np.zeros_like(dW)) + (1 - self.beta2) * dW**2
                    gradients["sdb" + str(i+1)] = self.beta2 * gradients.get("sdb" + str(i+1), np.zeros_like(db)) + (1 - self.beta2) * db**2
                    vdW_corrected = gradients["vdW" + str(i+1)] / (1 - self.beta1**self.t)
                    vdb_corrected = gradients["vdb" + str(i+1)] / (1 - self.beta1**self.t)
                    sdW_corrected = gradients["sdW" + str(i+1)] / (1 - self.beta2**self.t)
                    sdb_corrected = gradients["sdb" + str(i+1)] / (1 - self.beta2**self.t)
                    dW = vdW_corrected / (np.sqrt(sdW_corrected) + self.epsilon)
                    db = vdb_corrected / (np.sqrt(sdb_corrected) + self.epsilon)
                gradients["dW" + str(i+1)] = dW
                gradients["db" + str(i+1)] = db
        return gradients

    def update_parameters(self, gradients):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                layer.W -= self.learning_rate * gradients["dW" + str(i+1)]
                layer.b -= self.learning_rate * gradients["db" + str(i+1)]

    def train(self, X_train, y_train, epochs, learning_rate=0.1, batch_size=64, valid_split=0.2, optimizer=None, decay_rate=0):
        self.learning_rate = learning_rate
        fixed_lr = learning_rate
        self.optimizer = optimizer
        X_train, y_train, X_valid, y_valid = self.train_test_split(X_train, y_train, valid_split=valid_split)
        num_examples = X_train.shape[1]
        num_batches = num_examples // batch_size
        for epoch in range(epochs):
            train_cost = 0
            self.learning_rate = fixed_lr * np.exp(-decay_rate*epoch)
            for j in tqdm(range(0, num_examples, batch_size)):
                start = j
                end = min(j+batch_size, num_examples)
                X_batch = X_train[:, start:end]
                y_batch = y_train[:, start:end]
                output = self.forward(X_batch)
                cost = self.compute_cost(output, y_batch)
                train_cost += cost
                gradients = self.backward(y_batch, output)
                self.update_parameters(gradients)

            # Compute cost and accuracy
            test_cost = self.compute_cost(self.forward(X_valid), y_valid)
            train_cost /= num_batches
            _, train_acc = self.predict(X_train, y_train)
            _, test_acc = self.predict(X_valid, y_valid)
            self.costs.append(train_cost)
            self.test_costs.append(test_cost)
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            
            print(f'Epoch: {epoch+1}/{epochs}, train_cost: {train_cost:.5f}, test_cost: {test_cost:.5f}, train_acc: {train_acc:.5f}, test_acc: {test_acc:.5f}')
            
    def predict(self, X, y):
        m = X.shape[1]   # X = (768, m)
        # y = (26, m)
        y_hat = self.forward(X, training=False)
        p = np.argmax(y_hat, axis=0)
        y = np.argmax(y, axis=0)
        correct = np.sum(p == y)
        # print(f"train examples: {m}, correctly predicted: {correct}, accuracy: {correct/m}")
        return p, correct/m

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = (-1./m) * np.sum(Y * np.log(AL))
        cost = np.squeeze(cost)
        assert cost.shape == ()
        return cost
    
    def train_test_split(self, X, y, valid_split=0.2):
        # Number of examples
        num_examples = X.shape[1]
        # Splitting the data
        valid_split = int(num_examples * valid_split)
        X_train = X[:, valid_split:]
        y_train = y[:, valid_split:]
        X_valid = X[:, :valid_split]
        y_valid = y[:, :valid_split]

        return X_train, y_train, X_valid, y_valid
    
    def plot_cost(self):
        plt.plot(np.squeeze(self.costs))
        plt.plot(np.squeeze(self.test_costs))
        plt.legend(['train', 'test'], loc='upper right')
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate = " + str(self.learning_rate))
        plt.savefig('figures/cost.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_acc(self):
        plt.plot(np.squeeze(self.train_acc))
        plt.plot(np.squeeze(self.test_acc))
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title("Learning rate = " + str(self.learning_rate))
        plt.savefig('figures/acc.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath):
        self.cache.clear()
        self.costs.clear()
        self.train_acc.clear()
        self.test_acc.clear()
        self.test_costs.clear()
        for l in self.layers:
            if isinstance(l, DenseLayer):
                l.activation_function.activation_cache = None
            elif isinstance(l, DropoutLayer):
                l.mask = None
        gc.collect()
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
    
def extract_data_and_labels(dataset):
    # Initialize lists to store data and labels
    data = []
    labels = []

    # Loop through the dataset
    for image_tensor, label in tqdm(dataset):
        # Flatten the image tensor and convert to numpy array
        flattened_image = image_tensor.numpy().flatten()
        data.append(flattened_image)
        labels.append(label)

    # Convert lists to numpy arrays
    return np.array(data), np.array(labels)


def load_data(train=True):
    print("Loading dataset for " + ("training" if train else "testing") + "...")
    dataset = ds.EMNIST(root='./data',
                        split='letters',
                        train=train,
                        transform=transforms.ToTensor(),
                        download=True)
    X, y = extract_data_and_labels(dataset)
    
    X = X.T
    y = y - 1
    y_oh = np.eye(26)[y.astype('int32')]
    y_oh = y_oh.T
    return X, y_oh, y
    

if __name__ == "__main__":
    X_train, y_train, y_train_raw = load_data(train=True)
    X_test, y_test, y_test_raw = load_data(train=False)
    print(X_train.shape)
    # Now shuffle the training set
    m_train = X_train.shape[1]
    np.random.seed(1)
    permutation = list(np.random.permutation(m_train))
    X_train = X_train[:, permutation]
    y_train = y_train[:, permutation]
    y_train_raw = y_train_raw[permutation]
    
    np.random.seed(1)
    model = Network()
    model.add(DenseLayer(input_size=X_train.shape[0], output_size=512))
    model.add(DropoutLayer(rate=0.2))
    model.add(DenseLayer(input_size=512, output_size=128))
    model.add(DropoutLayer(rate=0.2))
    model.add(DenseLayer(input_size=128, output_size=128))
    model.add(DropoutLayer(rate=0.2))
    model.add(DenseLayer(input_size=128, output_size=64))
    model.add(DropoutLayer(rate=0.2))
    model.add(DenseLayer(input_size=64, output_size=64))
    model.add(DropoutLayer(rate=0.2))
    model.add(DenseLayer(input_size=64, output_size=26, activation='softmax'))
    # Assume X_train and y_train are defined
    model.train(X_train, y_train, epochs=60, batch_size=1024, learning_rate=0.0005, valid_split=0.15, optimizer='adam', decay_rate=0.08)
    
    model.plot_cost()
    model.plot_acc()
    
    y_hat, acc = model.predict(X_test, y_test)
    print(f'Test accuracy: {acc}')
    
    macro_f1 = f1_score(y_test_raw, y_hat, average='macro')
    print(f'Test macro F1 score: {macro_f1:.5f}')
    
    C_counts = confusion_matrix(y_test_raw, y_hat)
    C = confusion_matrix(y_test_raw, y_hat, normalize='true')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(C, 'Blues', vmax=0.1)
    # also show the numbers
    for i in range(26):
        for j in range(26):
            plt.text(j, i, f'{C_counts[i, j]}', horizontalalignment='center', verticalalignment='center', fontsize=8)
    plt.xticks(range(26), labels=[chr(i+97) for i in range(26)])
    plt.yticks(range(26), labels=[chr(i+97) for i in range(26)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# if __name__ != "main":
#     import torchvision.datasets as ds
#     import torchvision.transforms as transforms

#     # Load the EMNIST dataset for training and validation
#     train_validation_dataset = ds.EMNIST(root='./data', 
#                                         split='letters',
#                                         train=True,
#                                         transform=transforms.ToTensor(),
#                                         download=True)

#     # Load the EMNIST dataset for independent testing
#     independent_test_dataset = ds.EMNIST(root='./data',
#                                         split='letters',
#                                         train=False,
#                                         transform=transforms.ToTensor())
    

#     # Extract data and labels from the datasets
#     X_train_raw, y_train_raw = extract_data_and_labels(train_validation_dataset)
#     X_test_raw, y_test_raw = extract_data_and_labels(independent_test_dataset)

#     m_train = X_train_raw.shape[0]
#     m_test = X_test_raw.shape[0]
#     num_px = X_train_raw.shape[1]
    
#     X_train = X_train_raw.T
#     X_test = X_test_raw.T
#     y_train = y_train_raw - 1
#     y_test = y_test_raw - 1

#     y_train = np.eye(26)[y_train.astype('int32')]
#     y_test = np.eye(26)[y_test.astype('int32')]
#     y_train = y_train.T
#     y_test = y_test.T

#     # Now shuffle the training set
#     np.random.seed(1)
#     permutation = list(np.random.permutation(m_train))
#     X_train = X_train[:, permutation]
#     y_train = y_train[:, permutation]
#     y_train_raw = y_train_raw[permutation]
    
#     np.random.seed(1)
#     model = Network()
#     model.add(DenseLayer(input_size=X_train.shape[0], output_size=512))
#     model.add(DropoutLayer(rate=0.2))
#     model.add(DenseLayer(input_size=512, output_size=128))
#     model.add(DropoutLayer(rate=0.2))
#     model.add(DenseLayer(input_size=128, output_size=128))
#     model.add(DropoutLayer(rate=0.2))
#     model.add(DenseLayer(input_size=128, output_size=64))
#     model.add(DropoutLayer(rate=0.2))
#     model.add(DenseLayer(input_size=64, output_size=64))
#     model.add(DropoutLayer(rate=0.2))
#     model.add(DenseLayer(input_size=64, output_size=26, activation='softmax'))
#     # Assume X_train and y_train are defined
#     model.train(X_train, y_train, epochs=60, batch_size=1024, learning_rate=0.0005, valid_split=0.15, optimizer='adam', decay_rate=0.08)
    
#     model.plot_cost()
#     model.plot_acc()
    
#     y_hat, acc = model.predict(X_test, y_test)
#     print(f'Test accuracy: {acc}')
    
#     macro_f1 = f1_score(y_test_raw - 1, y_hat, average='macro')
#     print(f'Test macro F1 score: {macro_f1:.5f}')
    
#     C_counts = confusion_matrix(y_test_raw - 1, y_hat)
#     C = confusion_matrix(y_test_raw - 1, y_hat, normalize='true')
#     fig = plt.figure(figsize=(10, 10))
#     plt.imshow(C, 'Blues', vmax=0.05)
#     # also show the numbers
#     for i in range(26):
#         for j in range(26):
#             plt.text(j, i, f'{C_counts[i, j]}', horizontalalignment='center', verticalalignment='center', fontsize=8)
#     plt.xticks(range(26), labels=[chr(i+97) for i in range(26)])
#     plt.yticks(range(26), labels=[chr(i+97) for i in range(26)])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()