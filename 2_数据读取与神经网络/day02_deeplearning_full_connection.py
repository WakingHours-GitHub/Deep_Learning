# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

from tensorflow.examples.tutorials.mnist import input_data

# mnist = tf.keras.datasets.mnist(x_train, y_train)
# (x_test, y_test) = mnist.load_data()

tf.disable_v2_behavior()


def full_connection():
    """
    用全连接对手写数字进行识别
    注意, 该数据集,需要在site-packages\tensorflow中的
    那么你则可以看下你当前环境下的TensorFlow的example中是否有tutorials文件或是否有example文件夹
    然后将数据集复制进去
    1. 准备模型
    2. 构建模型
    3. 构造损失函数
    4. 优化损失
    :return:
    """
    # 1）准备数据, 导入数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    # 用占位符定义真实数据, 因为我们还不知道要抽取多少个样本, 所以这里先用占位符
    # 构建特征与标签
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 2）构造模型 - 全连接
    # [None, 784] * W[784, 10] + Bias = [None, 10]
    # 模型参数仍然用变量来表示, 给模型一些初始值(这里是随机给的)
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.01))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10], stddev=0.1))
    y_predict = tf.matmul(X, weights) + bias

    # 3）构造损失函数
    loss_list = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true)  # 返回损失列表(也就是每个样本的损失值)
    loss = tf.reduce_mean(loss_list)  # 求取平均值

    # 4）优化损失
    # optimizer = tf.traZin.GradientDescentOptimizer(learning_rate=0.01).minimize(loss) # 梯度下降
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # 5）增加准确率计算
    bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))  # 得到准确率

    # 初始化变量
    init = tf.global_variables_initializer()  # 显示的初始化变量操作, 我们还要再sess中开启它

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)  # 开启初始化变量的op
        # 我们将数据加载进来:
        # image, labels = mnist.train.next_batch(100)  # 一次加载进来100个图片
        # print(f"训练之前, 损失为{loss.eval()}")

        # 开始训练
        for i in range(5000):
            # 获取真实值
            image, label = mnist.train.next_batch(500)
            # 运行起来:
            _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={X: image, y_true: label})
            # 因为optimizer就是一个操作, 我们没必要接受他
            print("第%d次的损失为%f，准确率为%f" % (i + 1, loss_value, accuracy_value))
        # 训练完之后我们可以保存模型,


    """
    保存模型, 然后我们加载进来, 然后用测试集合进行测试:
    
    """
    return None


if __name__ == "__main__":
    full_connection()
