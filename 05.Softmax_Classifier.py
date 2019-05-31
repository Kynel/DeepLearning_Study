import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
tfe = tf.contrib.eager

# raw data
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# convert data into numpy and float format
x_data = np.asarray(x_data, dtype = np.float32)
y_data = np.asarray(y_data, dtype = np.float32)

NB_CLASSES = 3  # num of class

# check shape of data 
print(x_data.shape)
print(y_data.shape)

# Weight and bias setting
W = tfe.Variable(tf.random_normal([4, NB_CLASSES]), name = 'weight')
b = tfe.Variable(tf.random_normal([NB_CLASSES]), name = 'bias')
variables = [W, b]

# check varibles
print(W, b)

# def hypothesis function
def hypothesis(X):
    
    return tf.nn.softmax(tf.matmul(X, W) + b)

# def cost function
def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.log(logits), axis = 1)
    cost_mean = tf.reduce_mean(cost)
    
    return cost_mean

# def gradient function
def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)

        return grads

# def train function
def fit(X, Y, EPOCHS = 2000, VERBOSE = 100):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

    for i in range(EPOCHS):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % VERBOSE == 0):
            print('Loss at epoch %d: %f' %(i + 1, cost_fn(X, Y).numpy()))

# start training
fit(x_data, y_data)

# test
test_hypothesis = hypothesis(x_data)
print(tf.argmax(test_hypothesis, 1))
print(tf.argmax(y_data, 1))