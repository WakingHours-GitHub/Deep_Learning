import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt




def my_linear_regression():
    """
    自己实现一下线性回归的过程
    :return:
    """
    # 初始化变量:
    x = tf.random_normal(shape=[100, 1])
    y_true = tf.matmul(x, [[0.8]]) + 0.7 # 这里matmul是矩阵乘法

    # 我们使用变量保存模型参数, 方便调试, 并且变量可变
    weights = tf.Variable(
        tf.random_normal(shape=[1, 1])
    )
    bais = tf.Variable(
        tf.random_normal(shape=[1,1])
    )
    y_pred = tf.matmul(x, weights) + bais
    # 定义损失函数:
    error = tf.reduce_mean(tf.square((y_true - y_pred)))
    # 定义优化算法: 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    init = tf.global_variables_initializer()

    # 开启会话:
    with tf.Session() as sess:
        sess.run(init) # 要先运行, 才可以打印参数
        # 打印初始参数:
        print(f"初始参数: 权重: {weights.eval()}, 偏置: {bais.eval()}, 损失: {error.eval()}")

        # 开始训练:
        for i in range(10):
            sess.run(optimizer) # 注意这里要运行, optimizer
            print(f"第{i}次训练: 权重: {weights.eval()}, 偏置: {bais.eval()}, 损失: {error.eval()}")

            # 画出损失的图像
            plt.figure(0)
            plt.plot(i+1, float(error.eval()), '+')
            # plt.hold(True)
        plt.show()
    return None


if __name__ == '__main__':
    my_linear_regression()