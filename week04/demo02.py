#coding=utf-8
# 预测酸奶日销量y，x1和x2是影响日销量的两个因素。
# 应提前采集的数据有：一段时间内每日的x1因素，x2因素和销量y_.
# 在本例中用销量预测产量，最优的产量应该等于销量。由于目前没有数据集，所以模拟了一套数据集。
# 利用TensorFlow中函数水机生成x1,x2，制造标准答案y_ = x1+x2,为了更真实，求和后还加了正负0.05的随机噪声

#####自定义损失函数###########
#在实际生活中，往往在制造一盒酸奶的成本和销售一盒酸奶的利润是不等价的，因此，需要使用符合该问题的自定义损失函数
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
SEED = 23455
COST = 1 #酸奶的成本
PROFIT=9 #酸奶的利润

#生成模拟数据集；
rdm  =  np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)]for (x1,x2) in X]
#1.定义神经网络的输入、参数和输出，定义前向传播过程；
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)
#2.定义损失函数及反向传播方法
# loss_mse = tf.reduce_mean(tf.square(y_-y))
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) %32
        end = (i*BATCH_SIZE) %32 +BATCH_SIZE
        sess.run(train_step,feed_dict={x: X[start:end],y_: Y_[start:end]})
        if i % 500 == 0:
            print("After %d training steps,w1 is :" %i)
            print(sess.run(w1),"\n")
    print("Final w1 is :\n",sess.run(w1))




#最后观察发现w1的两个参数都要大于1，因为这里的成本低于利润，所以会多生产一些，以达到利润最大