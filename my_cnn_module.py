import numpy as np
import activation_function as af
import layer as ly
from dataset.mnist import load_mnist

class NeuralNetwork:
    def __init__(self):
        self.layers = {}
        self.last_layer = None

    def init_network(self):

        # Input Layer
        W = np.random.randn(16, 1, 3, 3) * np.sqrt(2 / 9) # He 초기값
        b = np.zeros(16)

        '''
        # MLP

        # Input Layer
        W1 = np.random.randn(784, 128) * np.sqrt(2 / 784) # He 초기값
        b1= np.zeros(128)
        # Hidden Layer 1
        W2 = np.random.randn(128, 64) * np.sqrt(2 / 128) # He 초기값
        b2 = np.zeros(64)
        # Hidden Layer 2
        W3 = np.random.randn(64, 10) * np.sqrt(2 / 64) # He 초기값
        b3 = np.zeros(10)

        self.layers['Flatten'] = ly.Flatten()
        self.layers['Affine1'] = ly.Affine(W1, b1)
        self.layers['ReLU1'] = ly.ReLU()
        self.layers['Affine2'] = ly.Affine(W2, b2)
        self.layers['ReLU2'] = ly.ReLU()
        self.layers['Affine3'] = ly.Affine(W3, b3)
        self.last_layer = ly.SoftmaxWithLoss()
        '''
        '''
        # Deep CNN
        self.layers['Conv1'] = ly.Conv2D(1, W, b)
        self.layers['Relu1'] = ly.ReLU()
        self.layers['Conv2'] = ly.Conv2D(2)
        self.layers['Relu2'] = ly.ReLU()
        self.layers['Pooling1'] = ly.MaxPool()
        self.layers['Conv3'] = ly.Conv2D(3)
        self.layers['Relu3'] = ly.ReLU()
        self.layers['Conv4'] = ly.Conv2D(4)
        self.layers['Relu4'] = ly.ReLU()
        self.layers['Pooling2'] = ly.MaxPool()
        self.layers['Conv5'] = ly.Conv2D(5)
        self.layers['Relu5'] = ly.ReLU()
        self.layers['Conv6'] = ly.Conv2D(6)
        self.layers['Relu6'] = ly.ReLU()
        self.layers['Pooling3'] = ly.MaxPool()
        self.layers['Flatten'] = ly.Flatten()
        self.layers['Affine1'] = ly.Affine(output_size=128)
        self.layers['Relu7'] = ly.ReLU()
        # Dropout
        self.layers['Affine2'] = ly.Affine(output_size=10)
        # Dropout
        self.last_layer = ly.SoftmaxWithLoss()
        '''

        # Simple CNN
        self.layers['Conv1'] = ly.Conv2D(1, W, b)
        self.layers['Relu1'] = ly.ReLU()
        self.layers['Pooling1'] = ly.MaxPool()
        self.layers['Conv2'] = ly.Conv2D(2)
        self.layers['Relu2'] = ly.ReLU()
        self.layers['Pooling2'] = ly.MaxPool()
        self.layers['Conv3'] = ly.Conv2D(3)
        self.layers['Relu3'] = ly.ReLU()
        self.layers['Flatten'] = ly.Flatten()
        self.layers['Affine1'] = ly.Affine(output_size=128)
        self.layers['Relu4'] = ly.ReLU()
        self.layers['Dropout'] = ly.Dropout()
        self.layers['Affine2'] = ly.Affine(output_size=10)
        self.last_layer = ly.SoftmaxWithLoss()

    def predict(self, x : np.array):
        
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward()
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

def get_test_data():
    # x는 문제, t는 정답
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten=False, one_hot_label=True)
    return x_train, x_test, t_train, t_test
    '''
    normalize = RGB값을 0.0 ~ 1.0 사이의 값으로 정규화하는 옵션입니다. (단순히 255로 나누어서 정규화합니다.)
    flatten = 이미지를 1차원 배열로 평탄화하는 옵션입니다. True로 설정하면 (60000, 784) 형태의 2차원 배열이 됩니다. 
    one_hot_label = 정답 레이블을 원-핫 인코딩 형태로 저장하는 옵션입니다. True로 설정하면 (60000, 10) 형태의 2차원 배열이 됩니다.
    60000 = 훈련 데이터의 수, 784 = 이미지의 픽셀 수 (28x28), 10 = 클래스의 수 (0~9)
    '''