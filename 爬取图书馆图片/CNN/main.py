import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用v2版本的行为

tf.logging.set_verbosity(tf.logging.ERROR)  # 忽略tf报错的等级
"""
Level	Level for Humans	        Level Description
0	        DEBUG	            all messages are logged (Default)
1	        INFO	            INFO messages are not printed
2	        WARN	            INFO and WARNING messages are not printed
3	        ERROR	            INFO, WARNING, and ERROR messages are not printed
"""
#
tf.reset_default_graph()  # 重置初始图
graph = tf.Graph()  # 属性

# 训练时候的, batch_size (批处理队列长度)
batch_size = 100

# 迭代次数
iter_num = 1000


# image information:
height = 50
width = 130
channel = 1


def read_picture(batch_size=100):
    """
    将文件从磁盘中读取进tf框架
    :return:
    返回读取好的批处理
    """
    # 构建文件名序列:
    path = "..\\checkout_gray"  # 识别代码需要保存的文件路径
    file_list = [os.path.join(path, filename) for filename in os.listdir(path)]
    # print(file_list)
    # 生成文件队列:
    file_queue = tf.train.string_input_producer(file_list)
    # 读取
    reader = tf.WholeFileReader()  # 实例化读取器
    key, value = reader.read(file_queue)
    # key就是文件名信息, 对应得就是标签值, 这里我们需要使用
    # value就是图片值
    # 解码
    value_decode = tf.image.decode_png(value)
    # print(value_decode) # Tensor("DecodePng:0", shape=(?, ?, ?), dtype=uint8)

    # 更新图片形状, 重置为图片形状, 因为图片未全部确定下来, 所以使用静态类型更新
    value_decode.set_shape(shape=[height, width, channel])
    # 修改类型为float32, 这样训练精度高
    value_cast = tf.cast(value_decode, dtype=tf.float32)
    # print(value_cast) # Tensor("Cast:0", shape=(50, 130, 3), dtype=float32)

    # 放入到批处理队列
    key_batch, value_batch = tf.train.batch(
        tensors=[key, value_cast],  # 需要批处理的tensor对象, 如果想要多个tensor对象, 可用列表框出来
        batch_size=batch_size,
        num_threads=2,
        capacity=200
    )
    # 结果
    # print(key_batch+"\n", value_batch)
    # Tensor("add:0", shape=(1,), dtype=string)
    # Tensor("batch:1", shape=(1, 50, 130, 3), dtype=float32)

    return key_batch, value_batch # 返回批处理队列


def convolutional_neural_network(x):
    # 输入图像x:[batch_size, 50, 130, 1]
    # 直接使用一层卷积图像
    # 卷积核: 3, 个数32, 步长:1
    # 激活层: 使用relu
    # 池化层: 核: 2, 步长:2

    # 卷积层
    with tf.variable_scope("convolutional"):
        # 定义filter和bias
        weights_conv_1 = tf.Variable(
            initial_value=tf.random_normal(shape=[3, 3, 1, 32], mean=0.0, stddev=2)
        )
        bias_conv_1 = tf.Variable(
            initial_value=tf.random_normal(shape=[32], mean=0.0, stddev=2)
        )

        x_conv_1 = tf.nn.conv2d(
            input=x,
            filter=weights_conv_1,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_conv_1

        # 激活层
        x_relu = tf.nn.relu(x_conv_1)

        # 池化层
        x_pool = tf.nn.max_pool(
            value=x_relu,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )

        # print(x_pool) # Tensor("convolutional/MaxPool:0", shape=(1, 25, 65, 32), dtype=float32, device=/device:GPU:0)

        # 全连接层
        # 改变输入图片形状, 使其修改成为矩阵(二阶张量)
        # 输入: x: [batch_size, 25, 65, 32] -> [batch_size, 25*65*32]
        # y_pred: [batch_size, 4*10]
        # x * weight + bias = y_true
        # -> weight:[25*65*32, 4*10], bias:[4*40]

        with tf.variable_scope("full_connection"):
            x_fc = tf.reshape(x_pool, shape=[-1, 25 * 65 * 32])
            weights_fc = tf.Variable(
                initial_value=tf.random_normal(shape=[25 * 65 * 32, 4 * 10], mean=0.0, stddev=1.0)
            )
            bias_fc = tf.Variable(
                initial_value=tf.random_normal(shape=[4 * 10], mean=0.0, stddev=1.0)
            )
            y_pred = tf.add(tf.matmul(x_fc, weights_fc), bias_fc)
            # print(y_pred)  # Tensor("convolutional/full_connection/Add:0", shape=(1, 40), dtype=float32, device=/device:GPU:0)

    return y_pred


"""
def convolutional_neural_network_2(x):
    # 输入图像：[batch_size, 50, 130, 1]
    # 卷积层：
    weight_conv_1 = tf.Variable(
        tf.random_normal(shape=[3, 3, 1, 32])
    )
    bias_conv_1 = tf.Variable(
        tf.random_normal(shape=[32])
    )
    x_conv_1 = tf.nn.conv2d(
        x,
        filter=weight_conv_1,
        strides=[1, 1, 1, 1], # 步长为1
        padding="SAME"
    ) + bias_conv_1

    # 激活层:
    x_relu_1 = tf.nn.relu(x_conv_1)

    # 池化层:
    x_pool_1 = tf.nn.max_pool(
        value=x_relu_1,
        ksize=[1, 2, 2, 1], # 池化核大小: 为2
        strides=[1, 1, 1, 1], # 步长为1
        padding="SAME"
    )

    # print(x_pool_1) # Tensor("MaxPool:0", shape=(?, 50, 130, 32), dtype=float32, device=/device:GPU:0)

    # 第二层卷积层:
    # 输入图像: (?, 50, 130, 32)
    weight_conv_2 = tf.Variable(tf.random_normal(shape=[5, 5, 32, 64]))
    bias_conv_2 = tf.Variable(tf.random_normal(shape=[64]))
    x_conv_2 = tf.nn.conv2d(
        input=x_pool_1,
        filter=weight_conv_2,
        strides=[1, 1, 1, 1],
        padding="SAME"
    ) + bias_conv_2
    x_relu_2 = tf.nn.relu(x_conv_2)
    x_pool_2 = tf.nn.max_pool(value=x_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # print(x_pool_2) # Tensor("MaxPool_1:0", shape=(?, 25, 65, 64), dtype=float32, device=/device:GPU:0)



    # 全连接层:
    # 传入图像[None, 25, 65, 64]
    # x * weight + bias = y_predict [None, 4*10]
    # -> weight:[25*65*64, 4*10] bias:[4*10]
    x_fc = tf.reshape(x_pool_2, shape=[-1, 25*65*64])
    weight_fc = tf.Variable(tf.random_normal(shape=[25*65*64, 4*10]))
    bias_fc = tf.Variable(tf.random_normal(shape=[4*10]))
    y_pred = tf.add(
        tf.matmul(x_fc, weight_fc),
        bias_fc
    )
    y_pred = tf.nn.relu(y_pred)

    y_pred = tf.nn.dropout(y_pred, keep_prob=0.5)

    return y_pred
"""


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


def CNN(image=None, is_train=True, is_load=True):
    # 在每次启动CNN前, 都要重置图.
    tf.reset_default_graph()  # 需要在网络结构之前加上重置图的操作，否则会报错

    # 读入图片数据:
    key_batch, image_batch = read_picture()

    # 准备模型
    with tf.device("/GPU:0"):  # use GPU:0
        with tf.variable_scope("model_prepare"):
            x = tf.placeholder(shape=[None, height, width, channel], dtype=tf.float32)
            y_true = tf.placeholder(shape=[None, 4 * 10], dtype=tf.float32)

        # 创建卷积神经网络模型
        y_pred = convolutional_neural_network(x)
        # y_pred = convolutional_neural_network_2(x)

        # 构造损失函数:
        # 因为是一个样本对应多个分类值, 所以不能使用softmax+交叉熵进行进行计算
        with tf.variable_scope("loss_function"):
            loss_list = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred
            )
            loss = tf.reduce_mean(loss_list)

        # 构造优化器
        with tf.variable_scope("optimizer"):
            learn_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

        # 计算准确率:
        # equal = tf.equal(
        #     tf.argmax(tf.reshape(y_true, shape=[batch_size, 4, 10]), axis=2),
        #     tf.argmax(tf.reshape(y_pred, shape=[batch_size, 4, 10]), axis=2)
        # ) [[False False False True]]

        equal_list = tf.reduce_all(
            tf.equal(
                tf.argmax(tf.reshape(y_true, shape=[-1, 4, 10]), axis=2),
                tf.argmax(tf.reshape(y_pred, shape=[-1, 4, 10]), axis=2)
            ),
            axis=1
        )  # all
        accuracy = tf.reduce_mean(tf.cast(equal_list, dtype=tf.float32))

        # 显示开启初始化变量
        with tf.variable_scope("variables_initializer"):
            init = tf.global_variables_initializer()

    # @ with tf.device("/GPU:0") 结束

    with tf.variable_scope("summay"):
        # 收集变量, 便于可视化
        tf.summary.scalar("loss", loss)  # 标量
        tf.summary.scalar("accuracy", accuracy)

        # 聚合我们收集的变量
        merged = tf.summary.merge_all()  # merged operation

    # 初始化保存器
    saver = tf.train.Saver()

    # 使用GPU
    with tf.device("/GPU:0"):
        result = None
        # 开启会话:
        with tf.Session() as sess:
            sess.run(init)  # 初始化变量
            # 线程协调员
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 开启队列线程

            # 创建事件文件, 用于在TensorBoard中进行可视化
            file_event = tf.summary.FileWriter("./event/model", graph=sess.graph)

            # 是否训练:
            if is_train:

                if is_load:  # 是否加载以前保存的模型
                    print("load model ing...")
                    if os.path.exists("checkpoint"):
                        saver.restore(sess=sess, save_path="./checkpoint/model.ckpt")
                    else:
                        print("model path not exist")

                ############################################################################
                session_learn_rate = 0.01
                for i in range(iter_num):  # 开始训练
                    key_value, image_value = sess.run([key_batch, image_batch])
                    key_value = key2filename(key_value)
                    # print(key_value)  # [[0 1 5 5]]

                    # 进行one_hot编码:
                    labels_value = tf.reshape(tf.one_hot(key_value, depth=10), shape=[-1, 4 * 10]).eval()
                    # print(labels_value) # [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0.
                    # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]


                    # 运行优化器:
                    _, loss_value, accuracy_value, summary = sess.run(
                        [optimizer, loss, accuracy, merged],
                        feed_dict={x: image_value, y_true: labels_value, learn_rate: session_learn_rate}  # 自定义学习率
                    )
                    print(f"{i + 1} train, loss:%10.6f, accuracy:%10.6f, learn_rate: %10.6f" % (loss_value, accuracy_value, session_learn_rate))
                    session_learn_rate *= ((iter_num-i)/iter_num)*0.0001

                    #
                    # summary = sess.run(merged, feed_dict={x: image_value, y_true: labels_value})
                    file_event.add_summary(summary, i)  # 将本次运行的summary添加早event中
                    if not (i % 100):  # 每100次保存一下代码
                        saver.save(
                            sess=sess,
                            save_path="./checkpoint/model.ckpt"
                        )
                        #



            else:  # 预测数据
                print("use CNN model to predict:")
                print("load model...")
                if os.path.exists("./checkpoint"):
                    saver.restore(sess=sess, save_path="./checkpoint/model.ckpt")
                else:
                    print("model path not exist")
                y_pred_value = sess.run(y_pred, feed_dict={x: image})
                result = tf.argmax(tf.reshape(y_pred_value, shape=[-1, 4, 10]), axis=2).eval()[0]
                # print("OCR result: ", result)
                # ''.join(str(result).strip('[]').split())
                # ''.join([str(x) for x in result.tolist()])

                result = ''.join([str(x) for x in result.tolist()])  # 直接返回的是str类型

                ################################################################################################

            coord.request_stop()
            # coord.should_stop()
            coord.join(threads=threads)

    return result


def get_image():
    # cv.imread("./tushuguan jiequ/resize_img.png")
    img = plt.imread("./tushuguan jiequ/resize_img.png")
    # matplotlib.image.imread()在读取图像的时候顺便归一化了。
    # 所以后面需要乘以255. -> uint8
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img_gray.shape)

    plt.imshow(img)
    plt.show()

    return np.array(np.reshape(img_gray, newshape=[1, height, width, channel])) * 255  # 重置


def conversion_image_format(img):
    return np.array(np.reshape(img, newshape=[1, height, width, channel]))  # 逆归一化


def image_test():
    root = "../picture"  # 根目录
    img_list = os.listdir(root)
    random_gary_list = random.sample(img_list, 6)
    # 画图
    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    for i in range(2):
        for j in range(3):
            img = cv.imread(root + '/' + random_gary_list[i * 3 + j], flags=cv.IMREAD_COLOR)
            # cv.imshow("img", img)
            # cv.waitKey(0)
            # print(img.shape)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ocr_result = CNN(conversion_image_format(img_gray), is_train=False, is_load=False)
            print("ocr_result: ", ocr_result)
            fig.add_subplot(2, 3, i * 3 + j + 1)
            plt.imshow(img[:, :, :: -1])
            # ocr result
            plt.title("OCR: " + ocr_result)
        #     break
        # break
    plt.savefig("./pic.png") # 保存图片
    plt.show()

    return None


if __name__ == '__main__':
    # 预测
    # img = get_image()
    # ocr_result = CNN(get_image(), is_train=False)
    # print(''.join(ocr_result.strip('[]').split()))
    image_test()
    #
    # 训练
    # CNN(is_train=True, is_load=True)
