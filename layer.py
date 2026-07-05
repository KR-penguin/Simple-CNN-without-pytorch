import activation_function as af
import numpy as np
from utils import im2col, col2im

class Node:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

class Affine(Node):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x : np.array):
        self.x = x.copy()
        return af.Affine(self.x, self.W, self.b)

    def backward(self, dout):
        # dout = 앞에서 넘어온 미분값
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

class ReLU(Node):
    def __init__(self):
        self.mask = None

    def forward(self, x : np.array):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
        
class SoftmaxWithLoss(Node):
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x : np.array, t : np.array):
        self.y = af.softmax(x)
        self.t = t
        self.loss = af.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1.0):
        batch_size = self.y.shape[0]
        dx = dout * (self.y - self.t)
        out = dx / batch_size
        return out

class Flatten(Node):
    def __init__(self):
        self.original_shape = None

    def forward(self, x : np.array):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)

class MaxPool(Node):
    def __init__(self, pool_h, pool_w, stride=1):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.x = None
        self.arg_max = None

    def forward(self, x : np.array):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, self.stride, 0)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        self.arg_max = np.argmax(col, axis=1) # 역전파를 위해 최대값의 인덱스 저장
        '''
        역전파때는 output에 영향을 미쳤던 데이터들만 역전파를 해줘야하므로,
        최대값만 쓰는 max pool 특성상 최대값 인덱스만 저장해두고 역전파때는 
        최대값이었던 인덱스에만 역전파를 해주는 방식으로 구현해야한다.
        '''
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # transpose는 (N, out_h, out_w, C) -> (N, C, out_h, out_w)로 차원 변경
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) # (N, C, out_h, out_w) -> (N, out_h, out_w, C)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size)) # dmax 초기화
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        '''
        # 1. 0으로 가득 찬 3행 4열짜리 빈 방 생성
        dmax = np.zeros((3, 4)) 
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]

        # 2. np.arange(3)은 [0, 1, 2]를 만듦 (행 번호)
        #    self.arg_max는 [2, 0, 3] (열 번호)
        #    즉, (0,2), (1,0), (2,3) 좌표를 콕 집음!
        dmax[[0, 1, 2], [2, 0, 3]] = [9.0, 8.0, 7.0]

        # 3. 결과
        # [[  0,   0, 9.0,   0],  <- 0번 행의 2번 열에 9.0 주입
        #  [8.0,   0,   0,   0],  <- 1번 행의 0번 열에 8.0 주입
        #  [  0,   0,   0, 7.0]]  <- 2번 행의 3번 열에 7.0 주입
        '''
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x_shape, self.pool_h, self.pool_w, self.stride, 0)
        '''
        dcol = (세로 : N * out_h * out_W, 가로 : C * pool_h * pool*w)
        out_h, out_w <= input_h, input_w
        일단 0으로 깔고 필요한 데이터만 넣는 형식으로 진행
        dx = (N, C, input_h, input_w)
        '''
        return dx

class Conv2D(Node):
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # (FN, C, FH, FW)
        self.b = b # 
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None # input x 행렬 펼처둔거
        self.col_W = None # weight W 행렬 펼처둔거
        self.dW = None # weight W 미분값
        self.dB = None # bias b 미분값

    def forward(self, x : np.array):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - FW) // self.stride + 1
        out_w = (W + 2 * self.pad - FH) // self.stride + 1
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = col @ col_W + self.b
        '''
        (세로, 가로)
        col = (N*OH*OW, C*FH*FW)
        col_W = (C*FH*FW, FN)
        col @ col_W = (N*OH*OW, FN)
        b = (FN)
        b -> broadcasting 일어남
        실제 연산할 때 b = (N*OH*OW, FN)

        따라서 out = (N*OH*OW, FN)
        '''
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        '''
        out = (N ,FN, out_h, out_w)
        '''
        self.x = x
        self.col = col  # backward를 위해서 저장
        self.col_W = col_W # backward를 위해서 저장
        return out


    def backward(self, dout : np.array):
        FN, C, FH, FW = self.W.shape
        # dout = (N ,FN, out_h, out_w)
        dout = dout.transpose(0, 2, 3, 1) # (N, out_h, out_w, FN)
        dout = dout.reshape(-1, FN) # (N*out_h*out_w, FN)
        self.db = dout.sum(axis=0)
        self.dW = self.col