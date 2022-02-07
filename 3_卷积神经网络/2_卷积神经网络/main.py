import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

# 是否记录操作和张量分配给哪些变量
# tf.debugging.set_log_device_placement(True)


train = True
# 训练时的数据集
batch_size = 200
if train == False:
    batch_size = 1


def read_picture():
    """
    读取验证码图片。
    :return:
    """
    # 1. 构造文件名队列
    # 以往多是通过os模块进行拼接， 这次我们使用一个新的方法
    # glob也可以进行文件名的处理
    file_list = [os.path.join("../../爬取图书馆图片/checkout", file) for file in os.listdir("../../爬取图书馆图片/checkout")]
    # print(file_list)
    # 创建文件名队列
    file_queue = tf.train.string_input_producer(file_list)  # 构建文件名队列

    # 2. 读取与解码
    # 读取
    reader = tf.WholeFileReader()  # 实例化读取器, 读取图片
    key, value = reader.read(file_queue)  # 从文件名队列中随机读取进来文件
    # 其中key是文件名字, value是文件值

    # 解码:
    image_decode = tf.image.decode_png(value)  # Tensor("DecodePng:0", shape=(?, ?, ?), dtype=uint8)
    # 此时的shape还是[?, ?, ?]

    # 更新图片形状, 静态. 更改静态形状: 在形状不完全确定下来的前提下, 也就是shape中出现?
    image_decode.set_shape([50, 130, 3])  # shape[height, width, channel]
    # print(image_decode) # Tensor("DecodePng:0", shape=(50, 130, 3), dtype=uint8)
    # 修改图片形状
    # 训练时使用float32, 提升精度
    image_cast = tf.cast(image_decode, dtype=tf.float32)
    # 查看处理过后的文件信息:
    print(image_cast)  # Tensor("Cast:0", shape=(50, 130, 3), dtype=float32)

    # 3. 构造批处理队列
    key_batch, image_batch = tf.train.batch(
        tensors=[key, image_cast],  # 放入批处理队列当中去
        batch_size=batch_size,
        num_threads=2,
        capacity=100
    )

    return key_batch, image_batch


def convolutional_neural_network(x):
    # 输入图像x: [batch_size, 50, 130, 3]

    # 1. 第一个大层
    # 输入图像: [batch_size, 50, 130, 3]
    # 包含卷积层, 激活层, 池化层
    # 卷积层: 卷积核大小:5, 个数:32, 步长: 1
    # 激活层: 使用relu
    # 池化层: 核:2 步长2,

    with tf.variable_scope("convolutional_1"):
        # 定义filtre和bias
        weights_conv_1 = tf.Variable(
            initial_value=tf.random_normal(shape=[5, 5, 3, 32], mean=0.0, stddev=0.1)
            # 创建的filter: [F, F, in_channel, out_channel(K)]
        )
        bias_conv_1 = tf.Variable(
            tf.random_normal(shape=[32], mean=0.0, stddev=0.1)
        )
        # 卷积层API
        x_conv_1 = tf.nn.conv2d(
            input=x,  # 输入变量
            filter=weights_conv_1,  # 卷积核
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_conv_1

        # 激活层
        x_relu_1 = tf.nn.relu(x_conv_1)

        # 池化层
        x_pool_1 = tf.nn.max_pool(
            value=x_relu_1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )
        # print(x_pool_1) # Tensor("convolutional_1/MaxPool:0", shape=(1, 25, 65, 32), dtype=float32)
        # 输出图像:
        # [batch_size, 25, 65, 32]

        # 2. 第二个大层
        # 输入图像: # [batch_size, 25, 130, 32]
        # 包含卷积层, 激活层, 池化层
        # 卷积层: 卷积核大小:5, 个数:64, 步长: 1
        # 激活层: 使用relu
        # 池化层: 核:2 步长2,
    with tf.variable_scope("convolutional_2"):
        # 卷积层
        weights_conv_2 = tf.Variable(
            tf.random_normal(shape=[5, 5, 32, 64], mean=0.0, stddev=0.1)
        )
        bias_conv_2 = tf.Variable(
            initial_value=tf.random_normal(shape=[64])
        )
        x_conv_2 = tf.nn.conv2d(
            input=x_pool_1,  # 输入变量
            filter=weights_conv_2,  # 卷积核
            strides=[1, 1, 1, 1],  # 步长
            padding="SAME"  # 使用跨越边缘取样
        ) + bias_conv_2
        # 激活层
        x_relu_2 = tf.nn.relu(x_conv_2)

        # 池化层
        x_pool_2 = tf.nn.max_pool(
            value=x_relu_2,  # 输入变量
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )
        # print(x_pool_2) # Tensor("convolutional_1/convolutional_2/MaxPool:0", shape=(1, 13, 33, 64), dtype=float32)
    # 输出图像: shape[1, 13, 33, 64]

    # 3. 全连接层
    # 这一层实际要做的是就是x * weight + bias = y_pred
    # 所以我们需要将x转换为矩阵, x[batch_size, 13*33*64], 最终得到的结果,y是一个[batch_size, 4*10] (四个样本, 每个样本取值是10个one-hot码)
    # x[batch_size, 13*33*64] * weight + bias = y_pred[batch_size, 4*10]
    # -> weight[13*33*64, 4*10], bias[4*10]

    # 改变输入图片形状, 使其修改成为矩阵(二阶张量)

    with tf.variable_scope("full_connection"):
        x_fc = tf.reshape(x_pool_2, shape=[batch_size, 13 * 33 * 64])
        weights_fc = tf.Variable(
            initial_value=tf.random_normal(shape=[13 * 33 * 64, 4 * 10], mean=0.0, stddev=0.1)
        )
        bias_fc = tf.Variable(
            tf.random_normal(shape=[4 * 10], mean=0.0, stddev=0.1)
        )
        y_pred = tf.add(tf.matmul(x_fc, weights_fc), bias_fc)

    # 验证其结果
    # print(y_pred)  #Tensor("convolutional_1/full_connection/Add:0", shape=(1, 40), dtype=float32)

    return y_pred


def key2filename(key_batch_value):
    # print(key_batch_value, "\n", type(key_batch_value))
    key_str = [str(key) for key in list(key_batch_value)]  # 将binary转换成str类型, 方便后面进行切割
    # print(key_str, type(key_str))
    filename_batch = [key[key.rfind("\\") + 1: key.rfind(".png")] for key in key_str]
    # print(filename_batch)  # ['9002', '9283']
    # 我们还需要将样本分割:
    filename_batch = break_labels(filename_batch)

    # 因为tf中都是ndarray类型的, 所以我们还需要将标签转换成ndarray类型
    return np.array(filename_batch)

    # for key in key_batch_value:
    #     print(str(key), type(str(key)))
    # 这里可以在ipython中尝试一下


def break_labels(filename_batch):
    break_labels_list = []
    for filename in filename_batch:
        temp = []
        for i in filename:
            # 还需要都转换成是整形
            temp.append(int(i))
        break_labels_list.append(temp)
    # print(break_labels_list)
    return break_labels_list


# def shift_picture():

def predict_picture():
    image = cv.imread("0003.png", 1)
    plt.imshow(image)
    plt.show()
    # print(image)
    image_nd = np.array(np.asarray(image, dtype=np.float32))
    print(image_nd.shape)
    image_nd = np.reshape(image_nd, newshape=[1, 50, 130, 3])
    print(image_nd)
    return image_nd


"""
def predict_picture():
    file_list = ['./0003.png']
    file_queue = tf.train.string_input_producer(file_list)
"""

if __name__ == '__main__':
    # 1. 读入图片数据
    key_batch, image_batch = read_picture()
    # print("key_batch", key_batch, "\nimage_batch", image_batch)  # 此时这两个张量都已经升阶了
    # key_batch Tensor("batch:0", shape=(200,), dtype=string)
    # image_batch Tensor("batch      :1", shape=(200, 50, 130, 3), dtype=float32)

    # 此时我们还需要将key_batch是带有路径和名字的一阶张量, 所以我们需要处理它
    # 使它变成与image相匹配的标签
    # 输入key_batch 返回 filename_batch
    # 因为需要key中的值, 所以需要开启会话, 所以我们在后面中进行这步

    # 构建模型:
    # 准备数据`
    # x: shape[batch_size, 50, 130 3]
    # y_true: shape[batch_size, 4*10] # 因为我们最终预测的结果只是数字, 然后一个样本对应多个类别
    with tf.variable_scope("data_preparation"):
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 50, 130, 3])
        y_true = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4 * 10])

    # 创建卷积神经网络
    # 我们需要创建网络架构, 返回y_pred
    y_pred = convolutional_neural_network(x)

    # 构造损失函数
    with tf.variable_scope("loss_function"):
        loss_list = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred
        )
        loss = tf.reduce_mean(loss_list)

    # 优化器
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    # 计算准确率
    # 就是比对y_true和y_pred
    equal_list = tf.reduce_all(
        tf.equal(
            tf.argmax(tf.reshape(y_pred, shape=[batch_size, 4, 10]), axis=2),
            tf.argmax(tf.reshape(y_true, shape=[batch_size, 4, 10]), axis=2)
            # tf.reshape(y_pred, shape=[batch_size, 4, 10]),
            # y_true
        ),  # 返回一个逻辑矩阵
        axis=1
    )

    # 得到的是逻辑行向量
    accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32))

    # 显示初始化变量
    init = tf.global_variables_initializer()

    # 初始化保存器
    saver = tf.train.Saver()

    # 开启会话:
    with tf.Session() as sess:
        # 运行初始化变量的op
        sess.run(init)

        # 线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 测试key2filename函数
        # key_batch_value = sess.run(key_batch)
        # label = key2filename(key_batch_value)
        # print("label", label)

        if True:  # 加载之前训练好的模型
            saver.restore(
                sess=sess,
                save_path="./checkpoint/model.ckpt"
            )

        # 开始训练:
        # 简洁的代码: 原理在下面
        if train:
            with tf.device('/GPU:0'):
                for i in range(1000):

                    key_value, image_value = sess.run([key_batch, image_batch])
                    labels = key2filename(key_value)
                    labels_finally = tf.reshape(tf.one_hot(labels, depth=10), shape=[batch_size, 4 * 10]).eval()
                    """
                    print("labels", labels)
                    plt.imshow(np.asarray(image_value[0], dtype=np.uint8))
                    plt.show()

                    
                    print("argmax reshape", np.argmax(np.reshape(labels_finally, newshape=[batch_size, 4, 10]), axis=2))
                    # print("labels", labels_finally)

                    y_pred_value = sess.run(y_pred, feed_dict={x: image_value})
                    print(y_pred_value)
                    y_pred_value = np.reshape(y_pred_value, newshape=[batch_size, 4, 10])
                    print(np.argmax(y_pred_value, axis=2))
                    """


                    # 运行优化器:
                    _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy],
                                                             feed_dict={x: image_value, y_true: labels_finally})

                    print(f"{i + 1} train, loss:{loss_value}, accuracy:%f" % accuracy_value)

                    if i % 100 == 0:
                        saver.save(
                            sess=sess,
                            save_path="./checkpoint/model.ckpt"
                        )
        else:  # 预测
            image = predict_picture()
            print(image)
            # image_test = sess.run(image) # 运行
            # 加载模型
            saver.restore(
                sess=sess,
                save_path="./checkpoint/model.ckpt"
            )
            y_pred_value = sess.run(y_pred, feed_dict={x: image})
            print("y_pred_value", y_pred_value)
            y_pred_value = np.reshape(y_pred_value, newshape=[4, 10])
            print(np.argmax(y_pred_value, axis=1))

        """
        # 原理:
        for i in range(1000):
            key_value, image_value = sess.run([key_batch, image_batch])
            labels = key2filename(key_value)
            # 但是label还是数字的形式, 所以我们需要转换成one-hot编码
            # 将标签值转换成one_hot编码
            # API: tf.one_hot() 返回值是原来的个数再加一个depth的维度, 改维度值就是depth
            # 如果indices是N维张量，那么函数输出将是N+1维张量,默认在最后一维添加新的维度).
            labels_value = tf.one_hot(
                indices=labels,
                depth=10,  # 深度, 也就是编码的范围.
            )  # 进行one_hot编码
            print(labels_value)  # Tensor("one_hot:0", shape=(2, 4, 10), dtype=float32)
            # 然后还需要改变维度, 把最后的dept融合, 跨阶 -> 动态调整形状
            labels_value = tf.reshape(labels_value, shape=[batch_size, 4 * 10])
            print(labels_value)  # Tensor("Reshape:0", shape=(2, 40), dtype=float32)
            # 此时形状就已经改变成我们理想的类型了.
            labels_value = labels_value.eval()  # 运行op

            # 运行优化器
            _, loss_value = sess.run([optimizer, loss], feed_dict={x: image_value, y_true:labels_value})
            print(f"{i+1} train: loss:{loss_value}")
        """

        coord.request_stop()
        coord.join(threads)
