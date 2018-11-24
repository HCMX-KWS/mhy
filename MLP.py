# TensorFlow搭建MLP
# 还有问题，待修改

import tensorflow as tf
import numpy as np

# 设置参数
n_sample = 1000
rate = 0.1
epoch = 50
display = 10
batch_size = 100

# 假装有数据
data = np.random.rand(n_sample, 500, 64)
label = np.random.randint(n_sample, size=1000)
input = []
output = []

for i in range(n_sample):
    input.append(np.reshape(data[i], [64*500]))
    onehot = np.zeros(1000, dtype=float)
    onehot[label[i]] = 1
    output.append(np.reshape(onehot, [1000]))

# 模型
n_input = 64 * 500
n_hidden = 128
n_output = 1000

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

Weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
bias = {
    'h': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

def model(x, Weights, bias):

    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, Weights['h']), bias['h']))
    out_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, Weights['out']), bias['out']))

    return out_layer

# 预测
pred = model(x, Weights, bias)

# 损失函数和优化策略
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

# 准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epoch):
        acc_total = 0
        total_batch = int(n_sample / batch_size)

        for i in range(total_batch):
            _, acc = sess.run([optimizer, accuracy], feed_dict = {x:input, y:output})
            acc_total += acc

        if((step + 1)% display == 0):
            print("Step = " + str(step + 1) + ", Accuracy = " +  "{:.2f}%".format(100 * acc_total / total_batch))
