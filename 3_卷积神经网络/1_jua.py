import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.disable_v2_behavior()

"""
使用全连接网络模型对mnist手写数字数据集进行分类

"""


def convolution_model(x):
    """
    使用卷积神经网络模型对模型进行抽取
    构建神经网络:
        有两个大层.
        每个卷积大层, 又分为卷积层, 激活层, 池化层
        最后有一个全连接层
    :param x: 输入值, 输入的图像. [None, 784]
    :return: y_pred ->表示预测的结果值
    """
    # 输入图形: shape[None 784]
    # 1. 第一个卷积大层:
    # 设置命名空间:
    with tf.variable_scope("convolution_model_1_tier"):
        # 处理图片形状
        # filter层需要的是4阶张量, 所以我们需要进行形状修改
        # [None, 784] -> [None, 28, 28, 1]跨越阶数的形状修改, 所以使用动态形状修改
        x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])  #
        # 因为reshape传入的参数必须是准确的,不能是None, 所以这里填-1
        # 大意是说，数组新的shape属性应该要与原来的配套，如果等于-1的话，
        # 那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
        # 如果是-1的话, 那么TF会根据总数/其他维度上的数字, 从而计算出来这该维度的数字

        # 定义卷积层 -> 其实就是定义filter和偏置
        conv1_weights = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 1, 32], mean=0.0, stddev=0.1)  # 定义filter的形状
        )
        conv1_bias = tf.Variable(
            initial_value=tf.random_normal(shape=[32], mean=0.0, stddev=0.1)
        )
        conv1_x = tf.nn.conv2d(
            input=x_reshape,  # 输入的张量
            filter=conv1_weights,  # filter张量, 表示选择的卷积核大小
            strides=[1, 1, 1, 1],  # 步长S
            padding="SAME"  # 表示零填充使用的算法,
            # 这里使用跨越边缘填充. 内部会自动选择零填充像素数, 所以不改变形状大小, 但还是要关心其他参数可能会改变形状大小
        ) + conv1_bias  # 加上偏置
        print("conv1_x", conv1_x)  # Tensor("convolution_model_1/add:0", shape=(?, 28, 28, 32), dtype=float32)

        # 激活层: -> 激活层不改变形状大小, 是为了增加非线性分割能力而设置的
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层:
        pool1_x = tf.nn.max_pool(
            value=relu1_x,  # 输入变量
            ksize=[1, 2, 2, 1],  # 池化窗口
            strides=[1, 2, 2, 1],  # 步长
            padding="SAME"
        )
        # 输出图像: 因为池化层为
        print("pool1_x:", pool1_x)  # Tensor("convolution_model_1/MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)/

    with tf.variable_scope("convolution_model_2_tier"):
        # 2. 第二个大层
        # 卷积层:
        conv2_weights = tf.Variable(
            tf.random_normal(shape=[5, 5, 32, 64], mean=0.0, stddev=0.1)
        )
        conv2_bias = tf.Variable(
            tf.random_normal(shape=[64], mean=0.0, stddev=0.1)
        )
        conv2_x = tf.nn.conv2d(
            input=pool1_x,
            filter=conv2_weights,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + conv2_bias

        # 激活层:
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层:
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        print("pool2_x: ",
              pool2_x)  # pool2_x:  Tensor("convolution_model_1/MaxPool_1:0", shape=(?, 7, 7, 32), dtype=float32)

    # 全连接层:
    with tf.variable_scope("full_connected_tier"):
        # 图像: [None, 7, 7, 64] -> [None, 7*7*64] 全连接层实际上就是矩阵相乘, 然后判断类别, 所以我们需要转换为2阶, 也就是矩阵
        # [None, 7*7*64] * weight + bias = [None, 10]
        # -> weight[7*7*64, 10] bias[10]
        # 处理图像:
        x_fc_reshape = tf.reshape(pool2_x, shape=[-1, 7 * 7 * 64])
        weights_fc = tf.Variable(
            tf.random_normal(shape=[7 * 7 * 64, 10], mean=0.0, stddev=0.1)
        )
        bias_fc = tf.Variable(
            tf.random_normal(shape=[10], mean=0.0, stddev=0.1)
        )
        # 最终计算得到y_pred
        y_pred = tf.add(
            tf.matmul(x_fc_reshape, weights_fc),
            bias_fc
        )
        return y_pred


def full_connected_mnist(istrain=True, img=None):
    """
    全连接神经网络模型
    单层全连接神经网络识别手写数字图片
    数据集介绍:
        这些图片都是黑白图片, 每张图片包含28 * 28像素.
        我们把这个数组展开为一个向量: 就是28*28 = 784.
    # 特征值: [None, 784]
    # 目标值: [None, 10] 因为是0~9类别, 所以最终的结果就是10的one-hot编码
    所以:
        x[None, 784] * weight + bias = y_pred[None, 10]
        infer:
            weight's shape: [784, 10]
            bias's shape: [10]

    :return:
    """
    # 数据准备
    mnist = input_data.read_data_sets("../2_数据读取与神经网络/mnist_data", one_hot=True)
    # x [None, 784] y_true [None. 10]
    with tf.variable_scope("data_prepare"):
        # 因为我们还不知道需要抽取多少个样本所以, 这里形状是None, 但是使用placeholder占位符
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_ture = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    # 2. 构建模型: 我们这里需要得出一个y_pred, 所以我们写一个函数, 用于构建我们的模型, 然后返回出来我们的y_predict
    y_pred = convolution_model(x)  #

    # 3. 定义损失函数
    with tf.variable_scope("softmax_crossentropy"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_ture,  # label
                logits=y_pred
            )
            # labels:真实值 [None, 10]  one_hot
            # logits:全脸层的输出[None,10]
            # 返回每个样本的损失组成的列表
        )  # 然后在用mean平均一下,得到最后的损失函数

    # 4. 选择优化器
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # 5. 计算准确率
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(
            tf.argmax(y_ture, 1),
            tf.argmax(y_pred, 1)
            # argmax()表示寻找最大值, 返回最大值的索引, axis = 1表示对二维为一个样本, 对一维度,进行操作
            # 也就是以列为一个样本, 计算行
        )
        # 此时equal_list是一组逻辑值, 所以我们还需要转换
        accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 模型的保存与加载:
    saver = tf.train.Saver()

    # 开启会话, 运行模型 (因为不涉及到什么批处理队列, 所以不需要开启线程)
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 开始训练:
        if istrain:
            with tf.device("/GPU:0"):
                for i in range(500):
                    # 获取数据, 实时提供
                    # 每步提供100个样本训练
                    mnist_x, mnist_y = mnist.train.next_batch(50)
                    # print(mnist_x.shape)
                    # run op
                    _, loss_value, accuracy_value = sess.run(
                        [optimizer, loss, accuracy],
                        feed_dict={x: mnist_x, y_ture: mnist_y}
                    )
                    print(f"train: {i + 1}, loss: {loss_value}, accuracy: {accuracy_value}")
                    if i % 100 == 0:
                        saver.save(
                            sess=sess,
                            save_path="./checkpoint/model.ckpt"
                            # save_path="./checkpoint/" + str(
                            #     datetime.datetime.now().strftime("%Y_%m_%d")),
                            # 注意, 父级路径必须存在, 否则save_path会报错, save_path只能创建1级文件夹和文件, 多级, 路径必须存在, 否则报错
                        )

        else:
            saver.restore(
                sess=sess,
                save_path="./checkpoint/model.ckpt"
            )
            result = sess.run(y_pred, feed_dict={x: img})
            print(result) # ndarray类型
            return result


    return None


def test():
    mnist = input_data.read_data_sets("../2_数据读取与神经网络/mnist_data", one_hot=True)
    image, label = mnist.test.next_batch(1)
    sc_image = image
    print(image)
    image = image * 100
    image_int = np.asarray(image, dtype=np.uint8)
    print("shape", image_int.shape)
    image_int = np.reshape(image_int, newshape=[28, 28, 1])
    # print(image_int)
    plt.imshow(image_int, cmap="gray")
    plt.show()

    return sc_image


if __name__ == '__main__':
    img = test()
    # full_connected_mnist()
    result = full_connected_mnist(istrain=True, img=img)
    print("识别结果:", np.argmax(result))

