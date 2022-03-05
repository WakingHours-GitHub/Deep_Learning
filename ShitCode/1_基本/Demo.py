import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def demo01():
    a_t = tf.constant(10)
    b_t = tf.constant(20)

    c_t = tf.add(a_t, b_t)
    print("c_t:", c_t)  # c_t: Tensor("Add:0", shape=(), dtype=int32)

    # 开启会话:
    sess = tf.Session()
    c_t_value = sess.run(c_t)
    print(c_t_value)
    sess.close()

    return None


if __name__ == '__main__':
    demo01()
