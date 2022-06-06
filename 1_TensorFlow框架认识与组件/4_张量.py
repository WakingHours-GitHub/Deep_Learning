import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def tensor_demo():
    """
    张量的演示

    :return:
    """
    tensor1 = tf.constant(4.0)  # -> 0阶张量
    vector = tf.constant([4, 1, 2, 3])  # 定义1阶张量
    matrix = tf.constant(
        [
            [1, 1],
            [2, 3]
        ],
        dtype=tf.float32
    )
    # 当我们使用op初始化张量的时候, 使用不同的shape就定义不同的形状, 也就是不同的阶数
    print("tensor1:", tensor1)  # tensor1: Tensor("Const:0", shape=(), dtype=float32)
    print("vector: ", vector)  # vector:  Tensor("Const_1:0", shape=(4,), dtype=int32)
    print("matrix: ", matrix)  # matrix  Tensor("Const_2:0", shape=(2, 2), dtype=float32)

    # 张量类型的修改
    vector_cast = tf.cast(vector, dtype=tf.float32)  # 不改变原来类型的张量, 返回转换后的新张量
    print("vector: ", vector)  # vector:  Tensor("Const_1:0", shape=(4,), dtype=int32)
    print("vector_cast: ", vector_cast)  # vector_cast  Tensor("Cast:0", shape=(4,), dtype=float32)
    # 可见, 并不改变原来的类型

    # 更新/改变静态形状
    # 静态形状: 只有在形状没有完全固定下来的情况下, 才可以通过tensor.set_shape()进行改变, 直接对原来的张量进行更改
    # 定义占位符
    # 没有完全固定下来的静态形状
    a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
    b_p = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    c_p = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    print("a_p", a_p)  # a_p Tensor("Placeholder:0", shape=(?, ?), dtype=float32)
    print("b_p", b_p)  # b_p Tensor("Placeholder_1:0", shape=(?, 10), dtype=float32)
    print("c_p", c_p)  # c_p Tensor("Placeholder_2:0", shape=(3, 2), dtype=float32)
    # 更新"形状未确定"的部分
    # 但是要注意, 已经定义的阶数仍然不能变, 是几阶就是几阶
    # 否则就会报错.

    a_p.set_shape([2, 3])  # a_p.set_shape([2, 3, 1]) # 报错
    b_p.set_shape([2, 10])  # 注意这里的10,已经固定了, 就不能修改了. #  b_p.set_shape([2, 5]) # 报错
    # c_p.set_shape(shape=[3, 1]) # 报错
    print("a_p", a_p)  # a_p Tensor("Placeholder:0", shape=(2, 3), dtype=float32) -> 更改
    print("b_p", b_p)  # b_p Tensor("Placeholder_1:0", shape=(2, 10), dtype=float32) -> 更改
    print("c_p", c_p)  # c_p Tensor("Placeholder_2:0", shape=(3, 2), dtype=float32) -> 没有更改
    # 所以只能更新形状未确定的部分!!!

    # 动态形状修改: 可以跨越维数进行转换, 并且与cast一样, 不改变原来的张量, 返回新的张量
    # 原则: 动态创建新张量时，张量的元素个数必须匹配(一致) !, 如果不一致, 则会报错, 跨阶也需要保持元素个一致.
    # 使用: def reshape(tensor, shape, name=None):  # pylint: disable=redefined-outer-name
    # 参数: 张量, 更改的形状, 指令名称
    a_p_re = tf.reshape(a_p, [3, 2]) # 将[2, 3] -> [3, 2]
    print("a_p_re: ", a_p_re) # a_p_re:  Tensor("Reshape:0", shape=(3, 2), dtype=float32) -> 成功
    print("a_p: ", a_p) # a_p:  Tensor("Placeholder:0", shape=(2, 3), dtype=float32) # 可见原来的没有被改变

    c_p_re = tf.reshape(c_p, shape=[2, 3, 1])
    print("c_p_re: ", c_p_re) # c_p_re:  Tensor("Reshape_1:0", shape=(2, 3, 1), dtype=float32)
    # c [3, 2] -> [2, 3, 1] 可见, 可以跨越阶数进行转换

    # c_p_re = tf.reshape(c_p, shape=[3, 3]) #
    # print(c_p_re) # 报错
    # 可见只有在转换前后元素相同时, 才可以进行reshape

    return None


if __name__ == '__main__':
    tensor_demo()
