import numpy as np
import layer as ly

N = 2
W = np.random.randn(8, 1, 28, 28) * np.sqrt(2/40000)
b = np.zeros(8)
conv1 = ly.Conv2D(1, W, b)
x = np.random.randn(N, 1, 28, 28)
out1 = conv1.forward(x)
print("conv1 out shape:", out1.shape)

relu1 = ly.ReLU()
out1 = relu1.forward(out1)

pool1 = ly.MaxPool()
try:
    out1p = pool1.forward(out1)
    print("pool1 out shape:", out1p.shape)
except Exception as e:
    print("pool1 error:", repr(e))
