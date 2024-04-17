import numpy as np
import matplotlib.pylab as plt
# 계단함수 구현하기

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

#################################################################

# def step_function(x):
#     y = x > 0
#     return y.astype(np.int_)

# x = np.array([-1.0, 1.0, 2.0])
# print("x : ", x)

# y = x > 0
# print("y : ", y)

# y = y.astype(np.int_)
# print("after y : ", y)

#################################################################

# def step_function(x):
#     return np.array(x > 0, dtype=np.int_)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()

#################################################################
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# x = np.array([-1.0, 1.0, 2.0])
# print("sigmoid", sigmoid(x))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) # y 축 설정
# plt.show()
#################################################################
# 3층 신경망
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def identity_function(x):
#     return x

# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#     return network

# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']

#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = identity_function(a3)
#     return y

# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)
#################################################################
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

# a = np.array([1010, 1000, 990])
# a_result = np.exp(a) / np.sum(np.exp(a))
# print(a_result)
# c = np.max(a)
# print(a - c)

#################################################################
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c) # 오버플로 대책
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

#################################################################
# MNIST
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle


#
#
# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#
# img = x_train[0]
# print("x_train : ", x_train)
# print("t_train : ", t_train)
# print("len x_train : ", len(x_train))
# print("len t_train : ", len(t_train))
# label = t_train[0]
# print(label)  # 5
#
# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
# print(img.shape)  # (28, 28)

# img_show(img)
#################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)

    p = np.argmax(y_batch, axis=1) # axis=1 : 최대값 인덱스 찾기.
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(accuracy_cnt)
print(len(x))
print("Accuracy:", str(float(accuracy_cnt)/len(x)))
#################################################################
