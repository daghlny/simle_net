
import numpy as np
import load_mnist as load
import os

def sigmoid(x):
    vals = 1/(1+np.exp(x))
    return vals

def softmax(x):
    max_x = np.max(x)
    vals = np.exp(x-max_x)
    vals = vals/np.sum(vals)
    return vals

def mean_squared_error(y, t):
    return np.sum((y-t)*(y-t))/2

def cross_entropy_error(y, t):
    delta = 1e-10
    return -np.sum(t*np.log(y+delta))

def cal_numerical_gradient(fx, x):
    h = 1e-4
    # here may be out-of-bound error
    x_shape = x.shape
    x = x.reshape(-1) 
    ret = np.zeros_like(x)
    for idx in range(x.size):
        tmp_x = x[idx]
        x[idx] = tmp_x+h
        f_1 = fx(x)
        x[idx] = tmp_x-h
        f_2 = fx(x)
        x[idx] = tmp_x
        delta = (f_1 - f_2)/ (2*h)
        ret[idx] = delta
    x = x.reshape(x_shape)
    ret = ret.reshape(x_shape)
    return ret

class SimpleNet:
    def __init__(self, input_size, hidden_size = 10, output_size = 10, study_rate = 0.01, is_debug=False):
        self.x_dim = input_size
        self.W1 = np.random.normal(0.5, 1, size=(hidden_size, input_size))
        self.b1 = np.random.normal(0.5, 1, size=(hidden_size))
        self.W2 = np.random.normal(0.5, 1, size=(output_size, hidden_size))
        self.b2 = np.random.normal(0.5, 1, size=(output_size))
        self.study_rate = study_rate
        self.is_debug = is_debug
        print("init simple net with w1={}, id(w1)={}, w2={}, id(w2)={}".format(self.W1.shape, id(self.W1), self.W2.shape, id(self.W2)))
    
    def predict(self, x):
        #print("x_dim={} w1={} b1={} w2={} b2={}".format(x.shape, self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape))
        y1 = sigmoid(np.dot(self.W1, x)+self.b1)
        y2 = softmax(np.dot(self.W2, y1)+self.b2)
        if self.is_debug:
            print("predict\nw1={} w2={} b1={} b2={} y1={} y2={}\n-------------------".format(self.W1, self.W2, self.b1, self.b2, y1, y2))
        return y2
    
    def update_params(self, w, gradients):
        return w+gradients*self.study_rate
    
    def study(self, x, label):
        # finish one iteration study
        w1_gradient = cal_numerical_gradient(lambda w: mean_squared_error(self.predict(x), label), self.W1)
        w2_gradient = cal_numerical_gradient(lambda w: mean_squared_error(self.predict(x), label), self.W2)
        b1_gradient = cal_numerical_gradient(lambda w: mean_squared_error(self.predict(x), label), self.b1)
        b2_gradient = cal_numerical_gradient(lambda w: mean_squared_error(self.predict(x), label), self.b2)
        if self.is_debug:
            print("gradient. w1={} w2={} b1={} b2={}\n---------------------".format(w1_gradient, w2_gradient, b1_gradient, b2_gradient))
        self.W1 = self.update_params(self.W1, w1_gradient)
        self.W2 = self.update_params(self.W2, w2_gradient)
        self.b1 = self.update_params(self.b1, b1_gradient)
        self.b2 = self.update_params(self.b2, b2_gradient)

    def dump_param(self):
        with open("./simple_net_params", "w") as f:
            f.write("w1={}\n\nw2={}\n\nb1={}\n\nb2={}\n\n".format(self.W1, self.W2, self.b1, self.b2))
        


def find_max_one(arr):
    idx = -1
    value = 0
    for i, a in enumerate(arr):
        if a > value:
            idx = i
            value = a
    return (idx, value)

def mnist_train():
    (train_data, train_labels), (test_data, test_labels) = load.load_mnist(one_hot_label=True)
    all = np.random.choice(train_data.shape[0], 30) # training times
    net = SimpleNet(train_data.shape[1], 100, train_labels.shape[1])
    i = 0

    for idx in all:
        print("{}/{} loop, idx={}".format(i, all.size, idx))
        net.study(train_data[idx], train_labels[idx])
        i += 1

    all = np.random.choice(test_data.shape[0], 10)
    net.dump_param()
    for idx in all:
        y = net.predict(test_data[idx])
        #print("y={} right={}".format(find_max_one(y), find_max_one(test_labels[idx])))

def tiny_train():
    train_data = np.array([np.array([1,0,1])])
    train_label = np.array([np.array([1,0,0])])
    net = SimpleNet(train_data.shape[1], 2, train_label.shape[1], is_debug=True)
    net.study(train_data[0], train_label[0])
    net.dump_param()


if __name__ == "__main__":
    mnist_train()



