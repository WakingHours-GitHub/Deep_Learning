# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

def tensorflow_demo():
    """
    初步使用TensorFlow
    认识TensorFlow基本结构
    :return:
    """
    print("tensorflow的版本: ", tf.__version__) # tensorflow的版本:  2.7.0

    # 原生python实现加法运算
    a = 2
    b = 3
    c = a + b
    print(f"原生加法运算:", c)

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果: ", c_t)

    # 不过Session已经在新版本的TensorFlow已经被删除了, 所以开启会话的操作也就没有了.
    # 不过我们可以使用一些操作, 使可以使用1.x版本:
    # import tensorflow._api.v2.compat.v1 as tf
    # tf.disable_v2_behavior()

    print("手动开启Session: \n")
    # 现在我们开启Session
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value", c_t_value)


    return None


if __name__ == '__main__':
    tensorflow_demo()
