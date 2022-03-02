# import tensorflow as tf
# 使用低版本:
import tensorflow._api.v2.compat.v1 as tf
import os
tf.disable_v2_behavior() # 禁用tf2_v2的行为

# 设置日志等级
# tf.logging.set_verbosity(tf.logging.FATAL) # 仅仅显示
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

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
    a_t = tf.constant(2) # 这就是创建了一个TF中的常量
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果: ", c_t)
    # Tensor("add:0", shape=(), dtype=int32) # 可以看到, 这仍然是一个Tensor对象, 而不是结果
    # 我们无法观察其值, 所以我们需要开启会话

    # 不过Session已经在新版本的TensorFlow已经被删除了, 所以开启会话的操作也就没有了.
    # 不过我们可以使用一些操作, 使可以使用1.x版本:
    # import tensorflow._api.v2.compat.v1 as tf
    # tf.disable_v2_behavior()

    print("手动开启Session: \n")
    # 现在我们开启Session
    # 我们可以查看各个操作都是使用什么设备去运行的
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        c_t_value = sess.run(c_t) # 运行起来
        print("c_t_value: ", c_t_value) # 输出结果


    return None

"""
TensorFlow结构分析:

"""


if __name__ == '__main__':
    tensorflow_demo()
