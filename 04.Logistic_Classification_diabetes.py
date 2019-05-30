# change Session version to eager execution
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility

print(tf.__version__)  # 1.13.1

# \Uxxxx may occur error! use /Uxxxx instead!
xy = np.loadtxt('C:/Users/unlea/OneDrive/Desktop/DeepLearning_Study-master/DeepLearning_Study/data-03-diabetes.csv', delimiter = ',', dtype = np.float32)
x_train = xy[0:700, 0:-1]
y_train = xy[0:700, [-1]]

x_test = xy[700:-1, 0:-1]
y_test = xy[700:-1, [-1]]

# check data and shape
print(x_train.shape, y_train.shape)
print(xy)

# use Tensorflow data API for data
BATCHSIZE = 700
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCHSIZE)

# Weight and Bias
W = tf.Variable(tf.zeros([8, 1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')

# def hypothesis, loss function and accuracy_check function
def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.02)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype = tf.float32))
    return accuracy

# calculate gradient by GradientTape
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
        return tape.gradient(loss_value, [W, b])

# train on Eager mode
EPOCHS = 10001

for step in range(EPOCHS):
    for features, labels in tf.contrib.eager.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars = zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Accuracy: {:.4f}, Loss: {:.4}".format(step, accuracy_fn(logistic_regression(features), labels), loss_fn(logistic_regression(features), features, labels)))

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))

'''
Iter: 0, Accuracy: 0.6571, Loss: 0.6916
Iter: 100, Accuracy: 0.6571, Loss: 0.6229
Iter: 200, Accuracy: 0.6557, Loss: 0.6034
Iter: 300, Accuracy: 0.6557, Loss: 0.5909
Iter: 400, Accuracy: 0.6600, Loss: 0.5806
Iter: 500, Accuracy: 0.6700, Loss: 0.5715
Iter: 600, Accuracy: 0.6843, Loss: 0.5634
Iter: 700, Accuracy: 0.6986, Loss: 0.5563
Iter: 800, Accuracy: 0.7129, Loss: 0.5498
Iter: 900, Accuracy: 0.7114, Loss: 0.5441
Iter: 1000, Accuracy: 0.7200, Loss: 0.5389
...
...
...
Iter: 8700, Accuracy: 0.7686, Loss: 0.4749
Iter: 8800, Accuracy: 0.7686, Loss: 0.4748
Iter: 8900, Accuracy: 0.7686, Loss: 0.4747
Iter: 9000, Accuracy: 0.7686, Loss: 0.4747
Iter: 9100, Accuracy: 0.7686, Loss: 0.4746
Iter: 9200, Accuracy: 0.7686, Loss: 0.4746
Iter: 9300, Accuracy: 0.7686, Loss: 0.4745
Iter: 9400, Accuracy: 0.7686, Loss: 0.4744
Iter: 9500, Accuracy: 0.7686, Loss: 0.4744
Iter: 9600, Accuracy: 0.7686, Loss: 0.4743
Iter: 9700, Accuracy: 0.7686, Loss: 0.4743
Iter: 9800, Accuracy: 0.7686, Loss: 0.4742
Iter: 9900, Accuracy: 0.7686, Loss: 0.4742
Iter: 10000, Accuracy: 0.7686, Loss: 0.4741
Testset Accuracy: 0.7931
'''