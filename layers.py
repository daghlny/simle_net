
import numpy as np
import net
import load_mnist


class Affine:
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.dx = None
        self.dw = None
        self.db = None

    def name(self):
        return "Affine"+str(self.w.shape)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) +self.b
    
    def backward(self, dy):
        self.dx = np.dot(dy, self.w.T)
        self.dw = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return self.dx
    
    def update_params(self, learning_rate=0.1):
        self.w = self.w - learning_rate*self.dw
        self.b = self.b - learning_rate*self.db

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def name(self):
        return "SoftMaxWithLoss"

    def forward(self, x, t):
        self.t = t
        self.y = net.softmax(x)
        #self.loss = net.cross_entropy_error(self.y, t)
        return self.y
    
    def backward(self):
        #print("y={}\nt={}\n".format(self.y, self.t))
        batch = self.t.shape[0]
        return (self.y-self.t)/batch
    
    def update_params(self):
        pass
    
class Relu:
    def __init__(self):
        self.x = None
        self.dx = None

    def name(self):
        return "Relu"

    def forward(self, x):
        self.x = x
        return np.where(x>0,x,0)
        
    def backward(self, dy):
        return np.where(self.x>0,dy,0)
    
    def update_params(self):
        pass
    
class SimpleNet:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.layers = []
        self.data_size = train_x.shape[0]
        self.input_size = train_x.shape[1]
        self.output_size = train_y.shape[1]
        print("data_size={} input_size={} output_size={}".format(self.data_size, self.input_size, self.output_size))
        self.layers.extend([Affine(self.input_size, 100), Relu(), Affine(100, self.output_size)])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x, t):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return self.lastLayer.forward(out, t)
    
    def predict_with_index(self, x, t):
        out = self.predict(x, t)
        return out, np.argmax(out, axis=1)
    
    def study(self, x, t):
        losses = self.predict(x, t)
        dy = self.lastLayer.backward()
        self.layers.reverse()
        for layer in self.layers:
            dy = layer.backward(dy)

        self.layers.reverse()
        for layer in self.layers:
            dy = layer.update_params()

    

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist.load_mnist(one_hot_label=True)
    n = SimpleNet(x_train[0:2], y_train[0:2], x_test, y_test)
    for i in range(0, x_train.shape[0], 10):
        print("loop i={}/{}".format(i, x_train.shape[0]))
        n.study(x_train[i:i+10], y_train[i:i+10])
    
    for i in range(0, 100, 10):
        print("--------------------------------")
        a, idx = n.predict_with_index(x_test[i:i+10], y_test[i:i+10])
        print("out={}\nrit={}\n".format(idx, np.argmax(y_test[i:i+10], axis=1)))