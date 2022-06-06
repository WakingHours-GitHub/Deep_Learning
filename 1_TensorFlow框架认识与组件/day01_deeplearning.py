import os

# import tensorflow as tf
# 我们使用tf2.7的版本, 但是教程中使用的是1.8, 于是我们做一些操作.
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志提醒等级
# 我们的cpu支持线性代数等匀速的加速, 但是我们使用的是pip安装, 所以就没有加载进来这种加速
# 这里就会有一个警告, 不过我们可以使用源码安装, 这样加速操作就可以用起来了..
# 或者设置一下日志等级. 这样就可以屏蔽掉一些警告


def tensorflow_demo():
    """
    TensorFlow的基本结构
    :return:
    """
    # 原生python加法运算
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果：\n", c)

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：\n", c_t)

    # 开启会话, 注: 新版本已经丢弃了.
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value:\n", c_t_value)

    return None


# import tensorflow.compat.v1 as tf    tf.disable_v2_behavior()    TensorFlow2版本的可以在导入时改为这两行代码，就可以当1.x
def graph_demo():
    """
    图的演示
    :return:
    """
    # TensorFlow实现加法运算, 实际上tf已经帮助我们建立了一个默认图.
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="a_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    # 查看一下
    print("a_t:\n", a_t) # Const:0
    print("b_t:\n", b_t) # Const_1:0
    # 其中Const表示指令名称,, 0表示只有一个输出, 所以为0, 若有两个输出,则为1,以此类推
    print("c_t:\n", c_t)
    # print("c_t.eval():\n", c_t.eval())

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph() # 获取默认图对象
    print("default_g:\n", default_g) # 打印内存地址


    # 方法2：查看属性, tensor的属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 自定义图
    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new", c_new)

        print("a_new:\n", a_new)
        print("b_new:\n", b_new)
        print("c_new:\n", c_new)
        print("a_new的图属性：\n", a_new.graph)
        print("c_new的图属性：\n", c_new.graph)

    # 开启会话
    with tf.Session() as sess:
        # c_t_value = sess.run(c_t)
        # 试图运行自定义图中的数据、操作
        # c_new_value = sess.run((c_new))
        # print("c_new_value:\n", c_new_value)
        # 因为Session(graph=None) # 默认指定的是默认图, 所以我们如果想要开启自定义图, 需要在graph指定
        #   def __init__(self, target='', graph=None, config=None):

        print("c_t_value:\n", c_t.eval())
        print("sess的图属性：\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
        # 可视化在tmp/summary中的文件, 并且保存的图形是该会话的graph

    # 开启new_g的会话
    with tf.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run((c_new))
        print("c_new_value:\n", c_new_value)
        print("new_sess的图属性：\n", new_sess.graph)

    return None


def session_demo():
    """
    会话的演示
    :return:
    """

    # TensorFlow实现加法运算
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="a_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    print("a_t:\n", a_t)
    print("b_t:\n", b_t)
    print("c_t:\n", c_t)
    # print("c_t.eval():\n", c_t.eval())

    # 定义占位符
    # def placeholder(dtype, shape=None, name=None):
    # 其中: dyype为类型.
    a_ph = tf.placeholder(tf.float32)
    b_ph = tf.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)
    # 查看是什么 -> 没有赋值形状, 所以只有指令对象, 和类型
    # 生成该变量的函数对象: Placeholder()
    print("a_ph:\n", a_ph)
    print("b_ph:\n", b_ph)
    print("c_ph:\n", c_ph)

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)

    # 方法2：查看属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 开启会话
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True)) as sess:
        # 运行placeholder
        # 在run的时候, 加上参数feed_dict, 字典类型: {张量: 值}
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 3.9, b_ph: 4.8})
        print("c_ph_value:\n", c_ph_value)
        # c_t_value = sess.run(c_t)
        # 试图运行自定义图中的数据、操作
        # c_new_value = sess.run((c_new))
        # print("c_new_value:\n", c_new_value)
        # 同时查看a_t, b_t, c_t
        a, b, c = sess.run([a_t, b_t, c_t]) # session.run()中的fetch参数, 可以传入一个列表,(返回一个列表) # 查看所有张量的属性.
        print("abc:\n", a, b, c)
        print("c_t_value:\n", c_t.eval())
        print("sess的图属性：\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)

    return None


def tensor_demo():
    """
    张量的演示
    :return:
    """
    tensor1 = tf.constant(4.0) # 0阶张量
    tensor2 = tf.constant([1, 2, 3, 4]) # 1阶张量
    linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)
    # 不同的shape -> 形状 -> 也就是阶数
    print("tensor1:\n", tensor1)
    print("tensor2:\n", tensor2)
    print("linear_squares_before:\n", linear_squares)

    # 张量类型的修改
    l_cast = tf.cast(linear_squares, dtype=tf.float32) # 改成float32类型.
    print("linear_squares_after:\n", linear_squares) # 原来的没有发生改变
    print("l_cast:\n", l_cast)  # 转换后的类型

    # 更新/改变静态形状
    # 定义占位符
    # 没有完全固定下来的静态形状
    a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
    b_p = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    c_p = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    print("a_p:\n", a_p)
    print("b_p:\n", b_p)
    print("c_p:\n", c_p)

    # 更新"形状未确定"的部分
    # 但是要注意, 已经定义的阶数仍然不能变, 是几阶就是几阶
    # 否则就会报错.
    # a_p.set_shape([2, 3])  # a_p.set_shape([2, 3, 1]) # 报错
    # b_p.set_shape([2, 10]) #  b_p.set_shape([2, 5]) # 报错
    # c_p.set_shape([2, 3])  # 报错
    # 所以只能更新形状未确定的部分!!!

    # 动态形状修改, 可以跨越维数进行转换
    a_p_reshape = tf.reshape(a_p, shape=[2, 3, 1])
    print("a_p:\n", a_p) # 本身没有被修改
    # print("b_p:\n", b_p)
    print("a_p_reshape:\n", a_p_reshape) # 修改成功

    # 但是需要保证, 改变前后的元素的数量相同
    c_p_reshape = tf.reshape(c_p, shape=[2, 3, 1])
    print("c_p:\n", c_p)
    print("c_p_reshape:\n", c_p_reshape)

    c_p_reshape = tf.reshape(c_p, shape=[3, 3]) # 此时就会报错.
    print("c_p:\n", c_p)
    print("c_p_reshape:\n", c_p_reshape)
    return None


def variable_demo():
    """
    变量的演示
    :return:
    """
    # 创建变量
    a = tf.Variable(initial_value=50)
    b = tf.Variable(initial_value=50)
    c = tf.add(a, b)
    # 查看属性
    print("a", a)
    print("b", b)
    print("c", c)
    # 但是在这开启会话的时候, 会报错.
    # 使用未经初始化的变量. 我们需要显示的初始化
    # 使用init = tf.global_variables_initializer()
    # 然后将init对象, 传给.tun方法中去



    # 添加命名空间
    with tf.variable_scope("my_scope"): # 起一个命名空间.
        a = tf.Variable(initial_value=50)
        b = tf.Variable(initial_value=40)
    with tf.variable_scope("your_scope"):
        c = tf.add(a, b)
    # 这样做, 可以将代码模块化, 这样结构比较清晰
    # 并且这样在TensorBoard中, 图的结构也会更加清晰
    # 打印.
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)

    # 初始化变量 -> 显示的初始化变量.
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init) # 放入刚才初始化的对象
        a_value, b_value, c_value = sess.run([a, b, c])
        print("a_value:\n", a_value)
        print("b_value:\n", b_value)
        print("c_value:\n", c_value)
        # 这样就输出成功了.
    return None




def linear_regression():
    """
    自实现一个线性回归
    完善这个线性回归模型
    :return:
    """
    with tf.variable_scope("prepare_data"):
        # 1）准备数据
        X = tf.random_normal(shape=[100, 1], name="feature") # 生成tensor, 并且生成name
        y_true = tf.matmul(X, [[0.8]]) + 0.7
        # 这里[[0.8]] -> 表示两个维度


        # 参数我们就是使用变量保存
    with tf.variable_scope("create_model"):
        # 2）构造模型
        # 定义模型参数 用 变量
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Weights") # 权重
        # Variable()还有一个参数: trainable=表示是否可以被训练. -> 用于设定某一个变量是否被训练.
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Bias") #
        y_predict = tf.matmul(X, weights) + bias # 构造模型

    with tf.variable_scope("loss_function"):
        # 3）构造损失函数
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope("optimizer"):
        # 4）优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 2_收集变量
    tf.summary.scalar("error", error) # 标量 -> (名字, Tensor)
    tf.summary.histogram("weights", weights) # 高维变量
    tf.summary.histogram("bias", bias)

    # 3_合并变量
    merged = tf.summary.merge_all()
    # 然后我们需要每次迭代都收集这个变量 -> 记录变量

    # 创建Saver对象 -> 用于保存, 加载模型.
    saver = tf.train.Saver()

    # 显式地初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 1_创建事件文件 -> TensorBoadr
        file_writer = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)

        # 查看初始化模型参数之后的值
        print("训练前模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))

        # 开始训练
        # for i in range(100):
        #     sess.run(optimizer) # op之间是有逻辑关系的
        #       opyimizer需要error, 而error需要y_predict,y_true, 于是数据就流动起来了 ->这就是TensorFlow
        #     print("第%d次训练后模型参数为：权重%f，偏置%f，损失为%f" % (i+1, weights.eval(), bias.eval(), error.eval()))
        #       我们需要不断的run()不断的训练

        #     # 运行: 合并变量操作
        #     summary = sess.run(merged) # 传入合并好的merged, 返回一个summary对象
        #     # 将每次迭代后的变量写入事件文件
        #     file_writer.add_summary(summary, i)
        #     # 开启TensorBoard
              # tensorboard --logdir="" # 到文件夹

        #     # 保存模型 sacer.save(sess, path)
        #     if i % 10 ==0: #  每十次保存一次模型
        #         saver.save(sess, "./tmp/model/my_linear.ckpt")

        # 加载模型
        if os.path.exists("./tmp/model/checkpoint"): # 先判断模型是否存在.
            saver.restore(sess, "./tmp/model/my_linear.ckpt")


        print("训练后模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))
        # 仍然为: 之前保存的模型.

    return None


# 1）定义命令行参数
tf.app.flags.DEFINE_integer("max_step", 100, "训练模型的步数")
tf.app.flags.DEFINE_string("model_dir", "Unknown", "模型保存的路径+模型名字")

# 2）简化变量名
FLAGS = tf.app.flags.FLAGS


def command_demo():
    """
    命令行参数演示
    :return:
    """
    # 3) FLAGS.前面定义的变量名
    print("max_step:\n", FLAGS.max_step)
    print("model_dir:\n", FLAGS.model_dir)

    return None


def main(argv): # 手动实现一个main函数, 并且如果想用tf.app.run()开启, 必须要加上(argv)
    print(argv) # 文件的路径
    # 没有这个argv就会报错.
    print("code start")
    return None


if __name__ == "__main__":
    # 代码1：TensorFlow的基本结构
    # tensorflow_demo()
    # 代码2：图的演示
    # graph_demo()
    # 代码3：会话的演示
    # session_demo()
    # 代码4：张量的演示
    # tensor_demo()
    # 代码5：变量的演示
    # variable_demo()
    # 代码6：自实现一个线性回归
    # linear_regression()

    # 代码7：命令行参数演示
    # command_demo() # 直接调用, 就是默认值
    # 命令行调用:
    # python 文件名.py --max_stop=200 --model_dir="hello world"

    tf.app.run()

