import tensorflow._api.v2.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


def full_connection():
    """
    使用一层神经网络, 全连接 对手写数字进行识别
    数据集: mnist 的手写字符数据集
    通过.train和.test分别调用训练集和测试集


    :return:

    """
    # 1. 导入数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)  # 加载数据对象
    """
    数据集介绍:
    数据集分为两部分: 55000行的训练数据集(mnist.train), 和10000行的测试数据集(mnist.test).
    每个数据单元有两部分组成: 一张包含手写数字的图片和一个对应的标签,
    我们把这些图片设为xs, 把这些标签设为ys, 训练集和测试集都包含xs和ys, 
    比如: 训练集的图片是xs, ys = mnist.train.image, 训练集的标签是xs, ys = mnist.train.label
    """

    # 2. 构建标签和特征
    # 我们需要使用占位符, 因为我们还不一定抽取多少个样本数
    #  placeholder提供占位符, run的时候通过feed_dict指定参数
    with tf.variable_scope("data_preparation"):  # 注意命名空间: ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$, 否则报错
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x")  # shape(样本数, 特征数) 因为是28*28=784, 所以特征数就784
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10],
                                name="y_true")  # 因为最后的结果是一个one-hot编码, 所以形状是:(样本数, 编码)
    print("x:", x)  # Tensor("data_preparation/x:0", shape=(?, 784), dtype=float32)
    print("y_true", y_true)  # Tensor("data_preparation/y_true:0", shape=(?, 10), dtype=float32)
    # 3. 构建全连接神经网络模型
    # x * weight + bias = y_pred
    # x[None, 784] * weight[784, 10] + bias[10] = y_pred[None, 10]
    # 其中weight, bias是模型的参数, 我们用变量保存 -> Variable() # 注意, 这里V需要大写
    # 增加命名空间:

    with tf.variable_scope("full_connection_model"):
        weights = tf.Variable(
            initial_value=tf.random_normal(
                shape=[784, 10],  # 形状
                mean=0.0,  # 均值
                stddev=0.01  # 标准差
            ),
            name="weights"
        )
        bias = tf.Variable(
            initial_value=tf.random_normal(
                shape=[10],  # 注意, 为一阶张量的时候, 必须使用[]
                mean=0.0,
                stddev=0.1
            ),
            name="bias"
        )
        y_pred = tf.add(tf.matmul(x, weights), bias)

    # 4. 构建损失函数: # 使用我们的交叉熵损失 a
    # API: tf.nn.softmax_cross_entropy_with_logits(labels, logits) -> 返回损失列表
    with tf.variable_scope("loss_function"):
        loss_list = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,  # 真实值
            logits=y_pred  # 预测值
        )  # 返回损失列表(也就是每个样本的损失值)
        loss = tf.reduce_mean(loss_list)
    print("loss_list",
          loss_list)  # Tensor("loss_function/softmax_cross_entropy_with_logits_sg/Reshape_2:0", shape=(?,), dtype=float32)
    print("loss", loss)  # loss Tensor("loss_function/Mean:0", shape=(), dtype=float32) 此时loss已经变成标量了, 也就是一个数值

    # 5. 优化损失, 这里使用的仍然是梯度下降算法:
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0.1,
    ).minimize(loss)  # 让loss取最小

    # 显示初始化变量操作:
    init = tf.global_variables_initializer()  # 在sess中开启这个操作, 此时所有的变量才可以使用

    # 添加准确率的计算:
    # 准确率如何计算,就是将的每个样本是否预测准确, 用0, 1表示, 求和起来, 除以总数
    bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)) # 返回一个bool列表
    # equal是计算, 前后两个列表是否相等的, 相等的位置为True, 不等的位置为False
    # 返回值是一布尔的列表, 但是我们还需要做计算, 所以转换成float32类型
    accuracy = tf.reduce_mean(tf.cast(bool_list, dtype=tf.float32))

    # 可视化;
    # 收集变量:
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("weights", weights)  # 高维变量
    tf.summary.histogram("bias", bias)
    # 聚合
    merged = tf.summary.merge_all()



    # 开启会话:
    with tf.Session() as sess:
        sess.run(init)  # 开启初始化变量的op
        # 创建事件文件:
        file_write = tf.summary.FileWriter("./event/temp", graph=sess.graph)

        # 将数据加载进来:
        images, labels = mnist.train.next_batch(1)  # 先加载进来100个图片, 为了查看初始损失值
        loss_value = sess.run(loss, feed_dict={x: images, y_true: labels})  # 启动一下
        # print(images)  # 查看是否加载进来了
        # 打印训练前的状态:
        print(f"训练之前, 损失为: {loss_value}")  # 训练之前, 损失为: 2.205940008163452

        # 绘图
        plt.figure(num=1, figsize=(16, 9), dpi=80)
        plt.plot(0, loss_value, marker="+")
        plt.text(0, loss_value, "begen")

        # 开始训练:1
        for i in range(1000):
            # 加载数据:
            images, labels = mnist.train.next_batch(600)
            _, loss_value, accuracy_new, summary = sess.run([optimizer, loss, accuracy,merged], feed_dict={x: images, y_true: labels})
            # print(f"经过第{i+1}次训练, 损失为: {loss_value}")
            print(f"train:{i + 1}, loss: {loss_value}, accuracy: {accuracy_new}")
            file_write.add_summary(summary, i)

            plt.plot(i+1, loss_value)
        plt.savefig("./img.png")
        plt.show()

    return None


if __name__ == '__main__':
    full_connection()
