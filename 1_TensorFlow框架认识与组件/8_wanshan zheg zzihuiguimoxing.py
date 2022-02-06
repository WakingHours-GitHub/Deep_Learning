"""


完善上一个自回归的模型

1. 增加变量显示:
    目的: 在TensorBoard当中观察模型参数, 损失值等变量值的变化
    1. 收集变量
        tf.summary.scalar(name='', tensor) 收集对于损失函数和准确率等单值变量 (标量)
            name为变量的名字, tensor为值
        tf.summary.histogrom(name='', tensor) 收集高维度的变量参数
        tf.summary.image(name='', tensor) 收集输入的图片张量能显示图片
    2. 合并变量写入事件文件
        merged = tf.summary.merge_all()
        运行合并: summary = sess.run(merged), 每次迭代都需要运行
        添加: FileWriter.add_summary(summary, i), 其中i表示第几次的值, 第几次迭代

2. 添加命名空间和指令名称
    使用:
    with tf.variable_scope():
        操作
    -> 可以让我们的TensorFlow在TensorBoard中显示更加清晰

3. 模型的保存与加载
    API:
        tf.train.Saver(var_list=None, max_to_keep=5)
            作用: 保存和加载模型 (保存文件格式: checkpoint文件)
            var_list: 指定将要保存和还原的变量. 它可以作为一个dict或者一个列表传递
            max_to_keep: 指示要保留的最近检查点文件的最大数量. 创建新文件时,会删除比较旧的文件
                        如果None或者0, 则保留所有检查点文件. 默认为5 (即保留最新的5个检查点文件)
    使用:
        1. 实例化Saver:
            saver = tf.train.Saver(var_list=None, max_to_keep=5)
        2. 保存:
            saver.save(sess, path)
            sess:为什么我们需要保存会话, 其实保存模型就是保存模型参数, 而模型参数就是数值,
                而只有才sess中run起来才能看到数值, 所以我们保存sess.
            path: 这里的path必须是: 指定目录+模型名字.后缀名 -> path/xxx.checkpoint
                并且目录必须存在, 否则报错
        3. 加载:
            saver.restore(sess, path)
            sess: 会话
            path: 需要加载模型的路径和模型名字

4. 命令行参数的使用



"""
import os

import tensorflow._api.v2.compat.v1 as tf
import datetime
import matplotlib.pyplot as plt
tf.disable_v2_behavior()


def linear_regression_batter():
    """
    完善上个案例的代码
    :return:
    """

    ## 添加命名空间
    # 数据:
    with tf.variable_scope("prepare_data"):
        x = tf.random_normal(shape=[100, 1], name="feature")  # generate a tensor
        y_true = tf.matmul(x, [[0.8]]) + 0.7  # 真实函数值
        # [[0.8]] 表示2阶, 方便与x相乘

    # 构建数据
    with tf.variable_scope("create_model"):
        # define model param by variable
        weights = tf.Variable(
            initial_value=tf.random_normal(shape=[1, 1], name="weights"),
            # trainable=None, # 该参数表示该变量是否需要被训练
        )
        bias = tf.Variable(
            initial_value=tf.random_normal(shape=[1, 1], name="bias")
        )
        y_pred = tf.matmul(x, weights) + bias  # 构造预测值.

    # 构造损失函数:
    with tf.variable_scope("loss_function"):
        # 均方误差:
        # MSE = avg((true-pred)^2)
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # 优化损失函数 -> 使用梯度下降:
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)  # 目标是loss最小

    # 2. 收集我们关注的变量
    tf.summary.scalar("loss", loss)  # 标量 -> 数值
    tf.summary.histogram("weights", weights)  # 高维变量
    tf.summary.histogram("bias", bias)

    # 集合(聚合)我们收集的变量
    merged = tf.summary.merge_all()  # 自动

    # 显示开启初始化变量
    init = tf.global_variables_initializer()

    # 保存模型: 实例化一个Saver对象
    save = tf.train.Saver()

    # 开启会话:
    with tf.Session() as sess:
        sess.run(init)  # 先运行
        # 然后查看初始值:
        print(f"初始状态: weights:{weights.eval()[0][0]}, bias:{bias.eval()[0][0]}, loss:{loss.eval()}")

        # 创建事件文件, 用于在tensorboard中可视化
        file_write = tf.summary.FileWriter("./event/temp", graph=sess.graph)

        # 开始训练:
        for i in range(100):
            sess.run(optimizer)  # 运行每次的优化器
            print(f"第{i}次训练: weights:{weights.eval()[0][0]}, bias:{bias.eval()[0][0]}, loss:{loss.eval()}")

            # 运行合并操作:
            summary = sess.run(merged)  # 上文定义好的merged, 在会话中运行起来, 返回summary对象
            # 然后将每次得到的结果, 添加进上面创建的事件文件: event
            file_write.add_summary(summary, i)  # 需要传入, summary对象和迭代第几次

            # 然后我们要保存模型, 肯定是要在训练中保存每次训练的模型啊
            # 所以需要在sess中的迭代中保存模型
            if i % 10 == 0: # 每10次保存一回
                # os.mkdir("./checkpoint/" + str(datetime.datetime.now().strftime("%Y_%m_%d")) + "/linear.checkpoint")
                save.save(
                    sess=sess, # , 参数是一个值, 而值只有在会话中运行起来, 才可以有值, 所以保存sess
                    save_path="./checkpoint/" + str(datetime.datetime.now().strftime("%Y_%m_%d")) + "/linear.checkpoint",
                    # 注意, 父级路径必须存在, 否则save_path会报错, save_path只能创建1级文件夹和文件, 多级, 路径必须存在, 否则报错
                )

            # 将损失描述出来:
            plt.figure(0)
            plt.plot(i-1, float(loss.eval()), 'k+')
            plt.title("loss and iterations")
            plt.xlabel('iterations')
            plt.ylabel('loss')
        plt.show()

    return None


if __name__ == '__main__':
    linear_regression_batter()
