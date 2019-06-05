import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
tfe = tf.contrib.eager

print(tf.__version__)

# x-> float32, y-> int32
# tf.one_hot function needs int input and tf.matmul function needs float input.
# so saperate x, y data
x = np.loadtxt('C:/Users/unlea/OneDrive/Desktop/DeepLearning_Study-master/DeepLearning_Study/data-04-zoo.csv', delimiter = ',', dtype = np.float32)
y = np.loadtxt('C:/Users/unlea/OneDrive/Desktop/DeepLearning_Study-master/DeepLearning_Study/data-04-zoo.csv', delimiter = ',', dtype = np.int32)
x_data = x[:, 0:-1]
y_data = y[:, [-1]]

NB_CLASSES = 7  # 0 ~ 6

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(list(y_data), NB_CLASSES)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NB_CLASSES])

print(x_data.shape, Y_one_hot.shape)  # (101, 16) (101, 7)

# weight and bias setting
W = tfe.Variable(tf.random_normal([16, NB_CLASSES]), name = 'weight')
b = tfe.Variable(tf.random_normal([NB_CLASSES]), name = 'bias')
variables = [W, b]

# tf.nn.softmax computes sotfmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def logits_fn(X):
  return tf.matmul(X, W) + b

def hypothesis(X):
  return tf.nn.softmax(logits_fn(X))

def cost_fn(X, Y):
  logits = logits_fn(X)
  cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y)
  cost = tf.reduce_mean(cost_i)

  return cost

def grad_fn(X, Y):
  with tf.GradientTape() as tape:
    loss = cost_fn(X, Y)
    grads = tape.gradient(loss, variables)
    
    return grads

def prediction(X, Y):
  pred = tf.argmax(hypothesis(X), 1)
  correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast -> 텐서를 새로운 형태로 캐스팅하는데 사용함

  return accuracy

# train function
def fit(X, Y, EPOCHS = 1000, VERBOSE = 10):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

  for i in range(EPOCHS):
    grads = grad_fn(X, Y)
    optimizer.apply_gradients(zip(grads, variables))
    if (i == 0) | ((i + 1) % VERBOSE == 0):
      acc = prediction(X, Y).numpy()
      loss = cost_fn(X, Y).numpy()
      print('Steps: {} Loss: {} Acc: {}'.format(i + 1, loss, acc))

fit(x_data, Y_one_hot)