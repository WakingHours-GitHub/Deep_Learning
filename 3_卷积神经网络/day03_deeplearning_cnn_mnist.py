# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

tf.disable_v2_behavior()

# 1、利用数据，在训练的时候实时提供数据
# mnist手写数字数据在运行时候实时提供给给占位符

# 这是tf的命令行参数
tf.app.flags.DEFINE_integer("is_train", 1, "指定是否是训练模型，还是拿数据去预测")
# 如果传入1 则训练模型, 如果是其他的我们就预测
FLAGS = tf.app.flags.FLAGS
# 在tensorflow中，tf.app.flags.FLAGS已经转移到tf.flags.FLAGS路径下
# tf.flags.FLAGS

def create_weights(shape):
    """
    专门定义一个产生随机权重的函数
    只需要传入形状即可
    :param shape: 形状
    :return:
    """
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))


def create_model(x):
    """
    构建卷积神经网络
    我们有两个卷积大层,
    每个卷积大层, 又分为: 卷积层, 激活层, 池化层
    最后有一个全连接层, 作为分类问题的输出
    :param x:
    :return:
    """
    # 1）第一个卷积大层
    with tf.variable_scope("conv1"):

        # 卷积层
        # filter层需要的是4阶张量, 所以我们需要进行形状修改
        # 将x[None, 784]形状进行修改
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 因为reshape传入的参数必须是准确的,不能是None, 所以这里填-1
        # 大意是说，数组新的shape属性应该要与原来的配套，如果等于-1的话，
        # 那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
        # 如果是-1的话, 那么TF会根据总数/其他维度上的数字, 从而计算出来这该维度的数字


        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5, 5, 1, 32]) # [F, F, in_channels, out_channels] -> [filter长, filter宽, 输入的通道数, 输出的通道数(卷积核个数)]
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias # 别忘了加上偏置

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, # 输入值
                                 ksize=[1, 2, 2, 1], # 池化窗口
                                 strides=[1, 2, 2, 1],  # 步长
                                 padding="SAME") #

    # 2）第二个卷积大层 根第一层差不多, 就是一些参数需要改变
    with tf.variable_scope("conv2"):

        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64]) # 形状改变, 输入的通道数 输出的通道数.
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        print("pool2_x: ", pool2_x) # pool2_x:  Tensor("conv2/MaxPool:0", shape=(?, 7, 7, 64), dtype=float32)

    # 3）全连接层
    with tf.variable_scope("full_connection"):
        # [None, 7, 7, 64]->[None, 7 * 7 * 64]
        # [None, 7 * 7 * 64] * [7 * 7 * 64, 10] + [10] = [None, 10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 7 * 7 * 64]) # 这里的-1, 仍然是让程序计算
        # 创建权重,
        weights_fc = create_weights(shape=[7 * 7 * 64, 10])
        bias_fc = create_weights(shape=[10])
        # 最终计算的值, 也就是预测值
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def full_connected_mnist():
    """
    单层全连接神经网络识别手写数字图片
    特征值：[None, 784]
    目标值：one_hot编码 [None, 10]
    :return:
    """
    mnist = input_data.read_data_sets("../2_数据读取与神经网络/mnist_data", one_hot=True)

    # 1、准备数据
    # x [None, 784] y_true [None. 10]
    with tf.variable_scope("mnist_data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 构建模型, 我们这里需要得出一个y_predict, 所以我们写一个函数.
    y_predict = create_model(x)

    # 3、softmax回归以及交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        # 亚当优化其
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        """
        TensorFlow中常见的优化器：
        GradientDescentOptimizer -> 梯度下降
        AdagradOptimizer
        AdagradDAOptimizer
        MomentumOptimizer
        AdamOptimizer  -> 自适应矩估计
        RMSPropOptimizer
        """

    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))


    # （2）收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # （3）合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # 加载模型
        # if os.path.exists("./tmp/modelckpt/checkpoint"):
        #     saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:
            # 循环步数去训练
            for i in range(3000):
                # 获取数据，实时提供
                # 每步提供50个样本训练
                mnist_x, mnist_y = mnist.train.next_batch(50)
                # 运行训练op
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                print("训练第%d步的准确率为：%f, 损失为：%f " % (i+1,
                                     sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                     sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})
                                     )
                  )

                # 运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)
                # if i % 100 == 0:
                #     saver.save(sess, "./tmp/modelckpt/fc_nn_model")

        else:
            # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                mnist_x, mnist_y = mnist.test.next_batch(1)
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                                                      i+1,
                                                      tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                                                      tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                                                      )
                                                      )

    return None


if __name__ == "__main__":
    full_connected_mnist()