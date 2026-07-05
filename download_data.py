import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=False, flatten=True, one_hot_label=False)
