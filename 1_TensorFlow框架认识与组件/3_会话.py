import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


# 会话的演示

def session_demo():
    """
    会话的演示.
    :return:
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = tf.add(a_t, b_t)
    print("a_t:\n", a_t)  # Tensor("Const:0", shape=(), dtype=int32)
    print("b_t:\n", b_t)  # Tensor("Const_1:0", shape=(), dtype=int32)
    print("c_t:\n", c_t)  # Tensor("Add:0", shape=(), dtype=int32)

    # 当我们没想好定义什么的时候, 我们可以使用placeholder
    # 定义占位符
    # def placeholder(dtype, shape=None, name=None):
    # 参数分别是: 类型, 形状, 和指令名称
    a_ph = tf.placeholder(tf.float32)
    b_ph = tf.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)
    # 查看是什么 -> 没有赋值形状, 所以只有指令对象, 和类型
    # 生成该变量的函数对象: Placeholder()
    print("a_ph:\n", a_ph)  # Tensor("Placeholder:0", dtype=float32)
    print("b_ph:\n", b_ph)  # Tensor("Placeholder_1:0", dtype=float32)
    print("c_ph:\n", c_ph)  # Tensor("Add_1:0", dtype=float32)
    # 我们使用placeholder() -> 操作函数 -> 内部实例化了一个placehlder操作对象 -> 对Tensor进行处理.
    # 所以第一个参数就变成了该placeholder对象

    # 开启会话: -> Session也是会使用一定的资源
    #  def __init__(self, target='', graph=None, config=None):
    # sess = tf.Session()
    # # 操作
    # sess.close()
    # 使用上下文管理器
    with tf.Session(
            # target='', # 当tf部署在多个机器上的时候, 可以指定目标地址
            graph=None,  # 指定运行的图为什么, None就是默认图
            config=tf.ConfigProto(  # 输出日志, 会话可以指定不同设备运行
                allow_soft_placement=True, #
                log_device_placement=True
            )
    ) as sess:
        # .run() 运行图
        # def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        # 作用: 通过使用sess.run() -> 来运行我们已经定义好的operation
        #   fetches: 单一的operation, 或者列表, 元组(其他不属于tf的类型不行)
        #   feed_dict: 参数允许调用者覆盖图中张量的值, 运行时赋值
        #       与tf.placeholder搭配使用, 则会检查值的形状是否与占位符兼容
        # 因为前面是使用placeholder, 所以这里我们run需要使用feed_dict=dict对变量进行赋值
        # 注意, 这里需要与前面的类型匹配
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 2, b_ph: 3})
        print("c_ph_value: ", c_ph_value) # c_ph_value:  5.0

        # 如果我们想要查看前面定义的张量, 我们需要run()或者eval -> abbr. 评估，评价（evaluation）
        # 同时查看a_t, b_t, c_t
        a_t_value, b_t_value, c_t_value = sess.run([a_t, b_t, c_t])
        print(f"a_t_value, b_t_value, c_t_value: {a_t_value}, {b_t_value}, {c_t_value}")
        # a_t_value, b_t_value, c_t_value: 2, 3, 5 -> 这样就得到了这些张量的值

        #  但是还有一种方法: eval()
        # 注意eval() 只能在Session开启中使用, 在外面使用, 则会报错.

        print("c_t.eval(): ", c_t.eval()) # c_t.eval():  5  --> 直接就可以查看


    return None


if __name__ == '__main__':
    session_demo()
