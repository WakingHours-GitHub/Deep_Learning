import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def variable_demo():
    """
    变量的演示。
    变量是专门存储模型参数的, 只要存储了模型参数我们就可以构建出来模型
    :return:
    """
    a = tf.Variable(initial_value=10)  # 设定初始值
    b = tf.Variable(initial_value=20)
    c = tf.add(a, b)
    # 查看属性:
    print("a", a)  # a <tf.Variable 'Variable:0' shape=() dtype=int32_ref>
    print("b", b)  # b <tf.Variable 'Variable_1:0' shape=() dtype=int32_ref>
    print("c", c)  # c Tensor("Add:0", shape=(), dtype=int32)
    # 此时a,b不是Tensor, 而是Variable -> 变量
    # 此时开启会话 -> 会报错, 原因是使用未经初始化的变量, 我们必须需要显示的初始化
    # with tf.Session() as sess:
    #     c_value = sess.run(c)

    # 显示初始化变量:
    init = tf.global_variables_initializer()  # 显示初始化所有变量
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)  # 启动init这个op, 而这个init op就是初始化所有的变量
        a_value, b_value, c_value = sess.run([a, b, c])  # 运行
        print("a_value", a_value)  # a_value 10
        print("b_value", b_value) # b_value 20
        print("c_value", c_value) # c_value 30
        # 这样就输出成功了.

    # 除此之外我们还可以为每个变量添加命名空间
    # 使用tf.variable_scope(name) 配合上下文管理器
    with tf.variable_scope("my_scope"):
        a = tf.Variable(initial_value=2)
        b = tf.Variable(initial_value=3)
    with tf.variable_scope("add_scope"):
        c = tf.add(a, b)
    # 打印属性:
    print("a", a) # a <tf.Variable 'my_scope/Variable:0' shape=() dtype=int32_ref>
    print("b", b) # b <tf.Variable 'my_scope/Variable_1:0' shape=() dtype=int32_ref>
    print("c", c) # c Tensor("add_scope/Add:0", shape=(), dtype=int32)
    # 此时, 命名空间就变了.
    # 这样做, 可以将代码模块化, 这样结构比较清晰
    # 并且这样在TensorBoard中, 图的结构也会更加清晰



    return None


if __name__ == '__main__':
    variable_demo()
