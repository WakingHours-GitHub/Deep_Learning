import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

"""
操作函数                        &                  操作对象
tf.constant(Tensor对象)            输入Tensor对象 -> 创建Const对象 -> 输出 Tensor对象
tf.add(Tensor对象1, Tensor对象2)   输入Tensor对象1, Tensor对象2 -> 经过Add对象 -> 输出 Tensor对象3
区分操作函数和操作对象

就是: 输入Tensor对象, 然后经过操作函数, 操作函数会实例化一个操作对象, 然后计算, 得到一个Tensor对象, 然后通过函数返回值返回该对象
        所以不管输入还是输出, 都是Tensor, 里面计算的细节被屏蔽了
        而打印Tensor对象, 一共有三个参数, :
        Tensor{"", shape=(), dtype=}
        第一个参数(我们称之为 指令名称 ) 就告诉你了这个Tensor变量是通过哪个操作对象得来的 -> 这个我们称之为指令名称
        
注意: 打印出来的是张量值, 可以理解成OP当中包含了这个值. 并且每一个op指令, 都可以对应一个唯一的值, 就是指令名称
    命名形式: <OP_NAME>:<int>
        <OP_NAME>: 是生成该张量的指令的名称
        <int>: 是一个整形, 它表示该张量在指令的输出中的索引
"""


def graph_demo():
    """
    TensorFlow中的图的应用:
    我们看一下graph的操作:
    :return:
    """
    # 还是那个例子: 加法
    a_t = tf.constant(2) # constant n. 常数，恒量；不变的事物
    b_t = tf.constant(3)
    c_t = a_t + b_t  # 传统的操作
    # right operation: 是通过操作函数来实现对Tensor进行操作
    c_t = tf.add(a_t, b_t)
    # 查看一下:
    print(a_t)
    print(b_t)
    print(c_t)
    # 输出:
    # Tensor("Const:0", shape=(), dtype=int32)
    # Tensor("Const_1:0", shape=(), dtype=int32)
    # Tensor("Add_1:0", shape=(), dtype=int32)
    # 我们也可以设置其指令名称:
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="d_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    # 查看一下:
    print(a_t)
    print(b_t)
    print(c_t)
    # Tensor("a_t:0", shape=(), dtype=int32)
    # Tensor("d_t:0", shape=(), dtype=int32)
    # Tensor("c_t:0", shape=(), dtype=int32)

    # 方法1: 查看默认图:
    default_graph = tf.get_default_graph()  # 获取默认图
    print("default graph:",
          default_graph)  # default graph: <tensorflow.python.framework.ops.Graph object at 0x00000218B79F2D90>

    # 方法2: 查看属性
    print("a_t的图属性:", a_t.graph)  # a_t的图属性: <tensorflow.python.framework.ops.Graph object at 0x000001DE729E6CA0>
    print("c_t的图属性:", c_t.graph)  # c_t的图属性: <tensorflow.python.framework.ops.Graph object at 0x000001DE729E6CA0>



    # 自定义图:
    my_graph = tf.Graph() # 实例化一个Graph对象
    # 在自己的图中定义数据和操作
    with my_graph.as_default():
        my_a_t = tf.constant(20)
        my_b_t = tf.constant(30)
        my_c_t = tf.add(my_a_t, my_b_t)
        print("my_c_t", my_c_t) # my_c_t Tensor("Add:0", shape=(), dtype=int32)
        # 同理对于my_a_t和my_b_t也是一样

        # 查看图属性
        print("my_c_t的图属性", my_c_t.graph) # my_c_t的图属性 <tensorflow.python.framework.ops.Graph object at 0x00000216E6F07250>

    # 开启Session
    with tf.Session() as sess:
        c_t_value1 = sess.run(c_t) # 执行c_t的操作

        # 看看sess的图属性：
        print("sess的图属性: ", sess.graph) # sess的图属性:  <tensorflow.python.framework.ops.Graph object at 0x0000019631F28FD0>

        # 尝试执行自定义图中的数据和操作
        # c_t_value2 = sess.run(my_c_t) # 报错
        # print("c_new_value:\n", c_new_value)
        # 因为Session(graph=None) # 默认指定的是默认图, 所以我们如果想要开启自定义图, 需要在graph指定
        #   def __init__(self, target='', graph=None, config=None):

        # 我们还可以将图写入本地生成event文件
        tf.summary.FileWriter("./event/tmp", graph=sess.graph)
        #   def __init__(self,
        #                logdir, # 保存的路径
        #                graph=None, # 指定的图
        #                max_queue=10,
        #                flush_secs=120,
        #                graph_def=None,
        #                filename_suffix=None,
        #                session=None):
        # 然后在cmd中使用:
        # tensorboard --logdir="path"
        # 在浏览器中打开TensorBoard的图页面,127.0.0.1:6006
        #             然后就可以看到可视化的效果了.asdasdddddddddddddddddddddddddddd

    # 开启my_graph的Session -> 需要指定名字
    with tf.Session(graph=my_graph) as my_sess:
        my_c_t_value = my_sess.run(my_c_t) # 运行
        print("my_c_t_value: ", my_c_t_value) # my_c_t_value: 50
        # 查看图属性:
        print("my_sess的图属性: ", my_sess.graph) # my_sess的图属性:  <tensorflow.python.framework.ops.Graph object at 0x000001425A6CCE50>

    return None


if __name__ == '__main__':
    graph_demo()
