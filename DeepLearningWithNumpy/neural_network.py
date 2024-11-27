import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

if __name__ == "__main__":
    # generate data
    num_samples = 100
    class_0 = np.random.normal(loc=[1, 1], scale=0.2, size=(num_samples, 2))
    labels_0 = np.zeros(num_samples)
    
    class_1 = np.random.normal(loc=[3, 3], scale=0.2, size=(num_samples, 2))
    labels_1 = np.ones(num_samples)

    data = np.vstack((class_0, class_1))
    labels = np.hstack((labels_0, labels_1)).flatten()

    
    # model
    fc1 = np.random.randn(2, 8)
    relu1 = relu
    fc2 = np.random.randn(8, 8)
    relu2 = relu
    fc3 = np.random.randn(8, 2)
    softmax1 = softmax

    
    

