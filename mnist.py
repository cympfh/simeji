import sys
import random
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer import Variable, optimizers, link, Chain
from load_mnist import load
import draw


def norm(x):
    """
    ベクトルの正規化
    x -> x/|x|
    """
    s = F.sum(x**2, axis=1) ** 0.5  # [a,b]^T
    height = s.data.shape[0]
    a = Variable(np.ones((1, height), dtype=np.float32))  # [1,1]
    eye = Variable(np.eye(height, dtype=np.float32))
    b = F.inv(F.matmul(s, a) * eye)  # [1/a, 0; 0, 1/b]
    return F.matmul(b, x)


def cos(a, b):
    """
    cos-similarity
    """
    height = a.data.shape[0]
    c = F.matmul(norm(a), norm(b), transb=True)
    eye = Variable(np.eye(height, dtype=np.float32))
    return F.sum(c * eye, axis=0)


class Encoder(Chain):
    def __init__(self):
        super().__init__(
            lin1=L.Linear(784, 100),
            lin2=L.Linear(100, 100),
            lin3=L.Linear(100, 100),
        )

    def __call__(self, x):
        h1 = F.dropout(self.lin1(x), train=True, ratio=0.2)
        h2 = F.relu(self.lin2(h1))
        y = norm(self.lin3(h2))
        return y


class Network(Chain):

    def __init__(self):
        self.enc = Encoder()
        self.enc.zerograds()
        self.opt = optimizers.SGD()
        self.opt.setup(self.enc)

    def sim(self, pairs):
        """
        pairs = [(a1, b1), (a2, b2), ...]
        sims = [s1, s2, ...]
        return sims
        """
        c1 = []
        c2 = []
        for a, b in pairs:
            c1.append(a)
            c2.append(b)
        c1 = Variable(np.array(c1, dtype=np.float32))
        c2 = Variable(np.array(c2, dtype=np.float32))

        x1 = self.enc(c1)
        x2 = self.enc(c2)

        ys = cos(x1, x2).data.tolist()
        return ys

    def train(self, pairs, sims):
        """
        pairs = [(a1, b1), (a2, b2), ...]
        sims = [s1, s2, ...]
        """

        c1 = []
        c2 = []
        for a, b in pairs:
            c1.append(a)
            c2.append(b)
        c1 = Variable(np.array(c1, dtype=np.float32))
        c2 = Variable(np.array(c2, dtype=np.float32))

        x1 = self.enc(c1)
        x2 = self.enc(c2)

        ys = cos(x1, x2)
        sims = Variable(np.array(sims, dtype=np.float32))
        loss = F.mean_squared_error(ys, sims)

        self.enc.zerograds()
        loss.backward()
        self.opt.update()

        return loss

net = Network()
data, test_data = load(100, 100)

for _ in range(1000):

    pairs = []
    sims = []

    for __ in range(70):  # batch

        idx1 = random.randrange(10)
        idx2 = (idx1 + random.randrange(9)) % 10
        i1 = random.randrange(len(data[idx1]))
        i2 = random.randrange(len(data[idx1]))
        i3 = random.randrange(len(data[idx2]))

        pairs.append((data[idx1][i1], data[idx1][i2]))
        sims.append(1.0)

        pairs.append((data[idx1][i1], data[idx2][i3]))
        sims.append(0.0)

    loss = net.train(pairs, sims)
    sys.stderr.write("\r# {} loss = {}".format(_, loss.data))
    sys.stderr.flush()


# test
for i in range(10):
    for j in range(len(test_data[i])):
        x = test_data[i][j]
        pairs = []
        for i2 in range(10):
            for j2 in range(len(test_data[i])):
                x2 = test_data[i2][j2]
                pairs.append((x, x2))
        sims = net.sim(pairs)
        print(' '.join(list(map(str, sims))))
