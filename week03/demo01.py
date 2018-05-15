#coding=utf-8
#0.导入模块，生成模拟数据集；
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed = 23455

#基于SEED生成随机数
rng = np.random.RandomState(seed)
#32行2列
X = rng.rand(32,2)

Y = [[int(x0 +x1 <1)] for (x0,x1) in X]

print("X:\n%s"%X)
print("Y:\n%s"%Y)

#1.定义神经网络的输入、参数和输出，定义前向传播过程；
x = tf.placeholder(tf.float32, shape=(None,2))  #输入
y_ = tf.placeholder(tf.float32, shape=(None,1))  #真实的结果
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) #第一层的参数 因为第一层是X为两列的，第二层隐含层设置为3列，所以这里random_normal是[2,3]
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1)) #第二层的参数

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#2.定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#3.生成会话，训练 STEPS 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("W1:\n",sess.run(w1))
    print("W2:\n",sess.run(w2))
    print("\n")

    STEPS=3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) %32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x: X[start:end],y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d traning step(s),loss on all data is %g"%(i,total_loss))
    print("\n")
    print("W1:\n",sess.run(w1))
    print("W2:\n",sess.run(w2))