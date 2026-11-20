import numpy as np

def ReLU(z):
    return np.maximum(0.0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def label_one_hot(y_true, num_labels):
    arr = np.zeros((y_true.shape[0], num_labels))
    for i, val in enumerate(y_true):
        arr[i, val] = 1
    return arr


class NeuralNetMLP:
    def __init__(self, num_classes: int, num_features: int, random_seed = 23) -> None: 
        super().__init__()
        self.num_classes = num_classes
        
        _rng = np.random.RandomState(random_seed)
        
        self.weight_layer1 = _rng.normal(loc = 0.0, scale=1.0, size= (num_features, 6))
        self.bias_layer1 = np.zeros(6)
        
        self.weight_layer2 = _rng.normal(loc = 0.0, scale=1.0, size= (6, 7))
        self.bias_layer2 = np.zeros(7)
        
        self.weight_out = _rng.normal(loc = 0.0, scale=1.0, size= (7, num_classes))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_h1 = np.dot(x, self.weight_layer1) + self.bias_layer1
        a_h1 = ReLU(z_h1)

        z_h2 = np.dot(a_h1, self.weight_layer2) + self.bias_layer2
        a_h2 = ReLU(z_h2)

        z_out = np.dot(a_h2, self.weight_out) + self.bias_out
        a_out = softmax(z_out)

        return a_out, a_h1, a_h2


    def backward(self, x, y, a_out, a_h1, a_h2):
        n_samples = x.shape[0]

        y_true_hot = label_one_hot(y_true=y, num_labels=self.num_classes)

        dL_dz_out = (a_out - y_true_hot) / n_samples

        dL_dW_out = np.dot(a_h2.T, dL_dz_out)
        dL_db_out = np.sum(dL_dz_out, axis=0)

        dL_da_h2 = np.dot(dL_dz_out, self.weight_out.T)

        dL_dz_h2 = dL_da_h2 * (a_h2 > 0)

        dL_dW_layer2 = np.dot(a_h1.T, dL_dz_h2)
        dL_db_layer2 = np.sum(dL_dz_h2, axis=0)

        dL_da_h1 = np.dot(dL_dz_h2, self.weight_layer2.T)

        dL_dz_h1 = dL_da_h1 * (a_h1 > 0)

        dL_dW_layer1 = np.dot(x.T, dL_dz_h1)
        dL_db_layer1 = np.sum(dL_dz_h1, axis=0)

        return {
            "dW_out": dL_dW_out,
            "db_out": dL_db_out,
            "dW_layer2": dL_dW_layer2,
            "db_layer2": dL_db_layer2,
            "dW_layer1": dL_dW_layer1,
            "db_layer1": dL_db_layer1
        }
