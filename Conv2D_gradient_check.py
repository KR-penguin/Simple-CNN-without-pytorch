from layer import Conv2D, SoftmaxWithLoss, Flatten, Affine
from utils import numerical_gradient
import numpy as np
BATCH_AMOUNT = 3
FILTER_AMOUNT = 2
CHANNEL_AMOUNT = 3
FILTER_HEIGHT = 3
FILTER_WIDTH = 3
PADDING = 1


def test():
    error = np.zeros(10)
    for x in range(10):
        W_small = np.random.randn(FILTER_AMOUNT, CHANNEL_AMOUNT, FILTER_HEIGHT, FILTER_WIDTH) * 0.1
        b_small = np.zeros(FILTER_AMOUNT)
        convolution_layer = Conv2D(W_small, b_small)

        x_small = np.random.randn(BATCH_AMOUNT, CHANNEL_AMOUNT, 27, 27)
        t_small = np.array([
            [0,0,0,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
        ])

        loss_layer = SoftmaxWithLoss()
        flaten = Flatten()
        affine = Affine()

        def loss_function():
            out = convolution_layer.forward(x_small) # out 차원 : (3, 2, 25, 25)
            out = flaten.forward(out) # out 차원 : (3, 1250)
            out = affine.forward(out) # out 차원 : (3, 10)
            return loss_layer.forward(out, t_small)

        loss_function()
        dout = loss_layer.backward()
        dout = affine.backward(dout)
        dout = flaten.backward(dout)
        dout = convolution_layer.backward(dout)


        numerical_gradient_weight = numerical_gradient(loss_function, convolution_layer.W)
        error[x] = np.abs(convolution_layer.dW - numerical_gradient_weight).max()
        print(error[x])

    print("\n\naverage : {0}".format(np.mean(error)))

test()