import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


#################################################################
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 2
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("2 일 확률 이 가장 높다 측정 : ", sum_squares_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print("7 이 확률 이 가장 높다 측정 : ", sum_squares_error(np.array(y), np.array(t)))

#################################################################
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
#################################################################
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import  load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(type(x_test))
print(x_test.shape)
print(t_test.shape)
#################################################################
