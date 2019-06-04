import TSDataSet

import tensorflow as tf

import numpy as np

batch_size=50


class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 300, 300, 1])#NHWC  a[None][28][28][1]
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,2])#a[None][10]

        #16个单通道卷积核卷单通数据,a[3][3][1][16],格式为：[height,width,in_channels, out_channels]
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6],stddev=0.1))#3*3的卷积核，1个输入通道，输出16个特征图(超参数)
        self.conv1_b = tf.Variable(tf.zeros(6)) #a[16]={0}

        #2个16通道卷积核卷16通道数据a[3][3][16][31],格式为：[height,width,in_channels, out_channels]
        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[2, 2, 6, 12],stddev=0.1))  # 3*3的卷积核，16个输入通道，输出32个特征图(超参数)
        self.conv2_b = tf.Variable(tf.zeros(12))#a[32]={0}

        self.conv3_w = tf.Variable(
            tf.truncated_normal(shape=[2, 2, 12, 24], stddev=0.1))  # 3*3的卷积核，16个输入通道，输出32个特征图(超参数)
        self.conv3_b = tf.Variable(tf.zeros(24))  # a[32]={0}

        self.conv4_w = tf.Variable(
            tf.truncated_normal(shape=[5, 5, 24, 48], stddev=0.1))  # 3*3的卷积核，16个输入通道，输出32个特征图(超参数)
        self.conv4_b = tf.Variable(tf.zeros(48))  # a[32]={0}

        self.w5 = tf.Variable(tf.random_normal(shape=[3 *3 * 48 , 2], dtype=tf.float32, stddev=0.02))
        self.b5 = tf.Variable(tf.zeros(shape=[2], dtype=tf.float32))  # 6*6*512
    def forward(self):
        #原 ,卷积操作
        self.bn = tf.contrib.layers.batch_norm(self.x, decay=0.9, updates_collections=None, epsilon=1e-5,
                                                  scale=True,
                                                  is_training=True)
        self.conv1 = tf.nn.leaky_relu(
            tf.nn.conv2d(input=self.bn, filter=self.conv1_w, strides=[1, 5, 5, 1], padding="SAME")+self.conv1_b)#28*28

        self.conv2 = tf.nn.leaky_relu(
            tf.nn.conv2d(self.conv1,self.conv2_w,[1,2,2,1],padding="SAME")+self.conv2_b)#14*14
        self.conv2 = tf.contrib.layers.batch_norm(self.conv2, decay=0.9, updates_collections=None, epsilon=1e-5,
                                                  scale=True,
                                                  is_training=True)
        self.conv3 = tf.nn.leaky_relu(
            tf.nn.conv2d(self.conv2, self.conv3_w, [1, 2, 2, 1], padding="SAME") + self.conv3_b)  # 14*14
        self.conv3 = tf.contrib.layers.batch_norm(self.conv3, decay=0.9, updates_collections=None, epsilon=1e-5,
                                                  scale=True,
                                                  is_training=True)
        self.conv4 = tf.nn.leaky_relu(
            tf.nn.conv2d(self.conv3, self.conv4_w, [1, 5, 5, 1], padding="SAME") + self.conv4_b)
        self.conv4 = tf.contrib.layers.batch_norm(self.conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                                                  scale=True,
                                                  is_training=True)
        # 14*14

        y2 = tf.reshape(self.conv4, [-1, 3 * 3 * 48])
        y2 = tf.nn.softmax(tf.matmul(y2, self.w5) + self.b5)
        self.out = y2

    def backward(self):
        self.loss = tf.reduce_mean((self.out-self.y)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    mydata = TSDataSet.MyDataset("JsamplePic", batch_size)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            xs,ys = mydata.get_bacth(sess)
            xs=np.reshape(xs,[-1,300,300,1])
            one_hot = tf.one_hot(ys, 2)
            ys=sess.run(one_hot)

            _loss,_,conv1 = sess.run([net.loss,net.opt,net.conv2],feed_dict={net.x:xs,net.y:ys})
            #print(type(conv1))
            #print(conv1.shape)

            if i % 10 == 0:
                print(_loss)