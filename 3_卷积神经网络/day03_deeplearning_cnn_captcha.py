# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
import glob
import pandas as pd
import numpy as np

tf.disable_v2_behavior()


# 1）读取图片数据filename -> 标签值
def read_picture():
    """
    读取验证码图片
    :return:
    """
    # 1、构造文件名队列
    # 以往都是通过os模块进行文件拼接, 这次我们使用一个新方法
    # glob也可以进行文件名的处理.
    file_list = glob.glob("./GenPics/*.jpg") # -> list
    # print("file_list:\n", file_list) # 返回值也是一个列表
    file_queue = tf.train.string_input_producer(file_list) # 构建文件名队列

    # 2、读取与解码
    # 读取
    reader = tf.WholeFileReader() # 实例化读取器, 读取图片
    filename, image = reader.read(file_queue) # read(文件名队列)
    # key, value = reader.read(file_queue)
    # key实际上就是文件名信息, value就是文件值信息
    # key需要找到真实标签名字, value需要解码成tensor中能识别的数据

    # 解码
    image_decode = tf.image.decode_jpeg(image)
    print("image_decode", image_decode) # shape:(?, ?, ?)

    # 更新图片形状, 静态更新 , 条件: 不完全确定下来的情况下.
    image_decode.set_shape([20, 80, 3]) # shape(height, width, channel)
    # print("image_decode:\n", image_dec·ode)
    # 修改图片类型
    image_cast = tf.cast(image_decode, tf.float32) # 训练的时候用float32, 提供精度
    # 并且使用卷积神经网络的时候, 必须是float32, 或者float64
    print("image_cast", image_cast) # shape(20, 80, 3) dtype=float32

    # 3、构造批处理队列
    filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=2, capacity=100)

    return filename_batch, image_batch

# 2）解析csv文件，将标签值NZPP->[13, 25, 15, 15]
def parse_csv():
    # 解析CSV文件, 建立文件名和标签值对应表格
    # 可以使用jupyter来看数据是如何变化的, 并且方便修改
    csv_data = pd.read_csv("./GenPics/labels.csv", names=["file_num", "chars"], index_col="file_num")

    # 根据字母生成对应数字
    # NZPP -> [13, 25, 15, 15]
    labels = [] # 二维列表
    for label in csv_data["chars"]:
        tmp = []
        for letter in label:
            tmp.append(ord(letter) - ord("A"))
        labels.append(tmp)

    csv_data["labels"] = labels # 添加一个字段
    # labels是二位列表, 里面元素都是一个列表

    return csv_data # 返回这个对象


# 3）将filename和标签值联系起来
def filename2label(filenames, csv_data):
    """
    将filename和标签值联系起来
    filename to label
    将一个样本的特征值和目标值一一对应
    通过文件名查表(csv_data)
    :param filenames:
    :param csv_data:
    :return:
    """
    # print("filename", filenames)
    labels = []

    # 将b'文件名中的数字提取出来并索引相应的标签值
    # 将key中的数字(编号) 抽取出来

    for filename in filenames:
        digit_str = "".join(list(filter(str.isdigit, str(filename))))
        # # filter(function, iterable) 第一个参数: 函数指针, 第二个参数: 需要判断的可迭代对象
        # 返回一个生成器.所以需要强转成list, 然后传换成str, 用join
        label = csv_data.loc[int(digit_str), "labels"]
        labels.append(label)

    # print("labels", labels)

    # print("labels:\n", labels)
    # 得到的就是本次批处理文件队列, 对应的标签值
    # 因为session开启的时候里面的类型都是ndarray, 所以我们这里使用np.array返回
    return np.array(labels)


# 4）构建卷积神经网络->y_predict
def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))

# 4）构建卷积神经网络->y_predict
def create_model(x):
    """
    构建卷积神经网络
    :param x:[None, 20, 80, 3]
    :return:
    """
    # 一定要注意形状的改变
    # 输入图像: x:[None, 20, 80, 3]
    # 1）第一个卷积大层
    with tf.variable_scope("conv1"):
        # 卷积层
        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5, 5, 3, 32]) # shape(F, F, in_channel, out_channel(K))
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias
        # 输出: [None, 20, 80, 32]
        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 输出图像[None, 10, 40, 32]
        # 第一层输出图像形状:shape[None, 10, 40, 32]
    # 2）第二个卷积大层
    with tf.variable_scope("conv2"):
        # [None, 20, 80, 3] --> [None, 10, 40, 32]
        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 同理, 输出形状:
        # [None, 5, 20, 64]

    # 3）全连接层
    with tf.variable_scope("full_connection"):
        # [None, 10, 40, 32] -> [None, 5, 20, 64] -> 结果
        # [None, 5, 20, 64] -> [None, 5 * 20 * 64]
        # 做全连接:
        # [None, 5 * 20 * 64] * [5 * 20 * 64, 4 * 26] + [4*26]  = [None, 4 * 26]
        # -> weights = [5*20*64, 4*26] -> bias = [4*26]
        x_fc = tf.reshape(pool2_x, shape=[-1, 5 * 20 * 64])
        weights_fc = create_weights(shape=[5 * 20 * 64, 4 * 26])
        bias_fc = create_weights(shape=[104])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict

# 5）构造损失函数
# 6）优化损失
# 7）计算准确率
# 8）开启会话、开启线程

if __name__ == "__main__":
    # 1.读入图片数据
    filename, image = read_picture()
    print(filename, image) # 就是批处理过后文件名和图片

    # 2. 解析csv文件
    csv_data = parse_csv()

    # 1、准备数据 ->
    # x: None, 20, 80, 3 -> 形状
    # y_true: None, 4*26, 计算损失, 我们需要是二维数组, 计算准确率需要的是4维数组
    x = tf.placeholder(tf.float32, shape=[None, 20, 80, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 4*26])

    # 2、构建模型
    y_predict = create_model(x)

    # 3、构造损失函数 -> 使用sigmoid交叉上损失
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list) # 求平均

    # 4、优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 5、计算准确率
    # y_true, y_pred都是[None, 4*26] 然后需要转换成三阶张量
    equal_list = tf.reduce_all(
        tf.equal(  # 比对真实值和预测值的范围.
            tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 26]), axis=2),
            tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2)
        ), # 返回一个逻辑矩阵
        axis=1  # 对列进行操作
        # (行, 列) axis = 1 计算列, 按照行, 返回列列向量
    )
    # 得到的是逻辑行向量
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32)) # 计算准确率

    # 初始化变量, Variable() 变量
    init = tf.global_variables_initializer() #


    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init)

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            # print("filename_value:\n", filename_value)
            # print("image_value:\n", image_value)

            labels = filename2label(filename_value, csv_data) # 拿到labels,
            # print(labels) # [[ 8 22 10  9]
            # 但是labels还是数字的形式, 所以我们需要转化成one_hot编码
            # 将标签值转换成one-hot:
            # API:tf.one_hot() 返回值是原来个数在加一个depth的维度, 改维度值就是depth
            # 在ipython中验证一下 # 开启交互式模式会话: tf.InteractiveSession()
            labels_value = tf.reshape(tf.one_hot(labels, depth=26), [-1, 4*26]).eval()
            _, error, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={x: image_value, y_true: labels_value})
            # 展示数据：



            print("y_Ture:", labels_value)
            print("y_pred", sess.run(y_predict, feed_dict={x: image_value}))




            print("第%d次训练后损失为%f，准确率为%f" % (i+1, error, accuracy_value))

        # 回收线程
        coord.request_stop()
        coord.join(threads)