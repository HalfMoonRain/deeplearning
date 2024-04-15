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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.array([-1.0, 1.0, 2.0])
print("sigmoid", sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y 축 설정
plt.show()
#################################################################