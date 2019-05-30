import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.eager as tfe  # error in VsCode but works!!

tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
print(tf.__version__)

# x_data is two-dimensional-array
# test data is red dot in graph
x_train = [[1., 2.],
           [2., 3.],
           [3., 1.],
           [4., 3.],
           [5., 3.],
           [6., 2.]]
y_train = [[0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1, x2, c = colors, marker = '^')
plt.scatter(x_test[0][0], x_test[0][1], c = "red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# use Tensorflow data API for data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# Weight and Bias (can be 0 or random(tf.random_normal([2,1])))
W = tf.Variable(tf.zeros([2, 1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')

# def hypothesis, loss function and accuracy_check function
def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype = tf.int32))
    return accuracy

# calculate gradient by GradientTape
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
        return tape.gradient(loss_value, [W, b])

# train on Eager mode
EPOCHS = 1001

for step in range(EPOCHS):
    #for features, labels in tfe.Iterator(dataset):  # error in VsCode but works!!
    for features, labels in tf.contrib.eager.Iterator(dataset):   
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features), features, labels)))

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))
