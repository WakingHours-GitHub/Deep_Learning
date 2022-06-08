import os

import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def read_picture_demo():
    """
    使用tf读取狗的图片案例.

    步骤:
    1. 构造文件名字序列
        是路径+文件名的列表: [path + dir_name, ...]
    2. 读取与解码
        读取: 从文件队列中读取
        解码: 解码成tf中的Tensor变量
        图片特征处理
    3. 批处理队列
    :return: None
    """
    # 1. 构造文件名队列 # string_input_producer("string_tensor")
    # 准备文件列表
    # print(os.listdir("dog")) # 读取文件夹下的所有文件
    # file_names = os.listdir("dog")
    # file_list = [os.path.join("./dog/", file_name) for file_name in file_names]
    file_list = [os.path.join("./dog/", file_name) for file_name in os.listdir("dog")]
    # print(file_list)
    file_queue = tf.train.string_input_producer(  # Queue_runner对象, 需要开启线程
        string_tensor=file_list,  # 路径 + 文件名字 的列表s
        shuffle=True,  # 是否随机读取到 文件队列 中
        num_epochs=None  # 过几遍数据, 默认(None)过无限次数据
    )  # -> 返回文件名队列
    # string_tensor: 需要的是一阶张量, 但是我们说过, 只需要传入文件列表就可以, 底层会默认转换成列表
    # 返回的就是 文件名队列 -> file_queue
    # 但是这是一个文件队列, 我们需要开启线程他才可以工作

    # 2. 读取与解码
    # 读取:
    reader = tf.WholeFileReader()  # 返回一个读取器对象
    key, value = reader.read(file_queue)  # 读取器都有一个reader()方法
    # 返回的key就是文件名字, value就是读取的图片的原始编码形式
    print("key:", key)  # key: Tensor("ReaderReadV2:0", shape=(), dtype=string)
    print("value:", value)  # value: Tensor("ReaderReadV2:1", shape=(), dtype=string)
    # 类型都是string, 其实也就是图片的字节码, 用字符串保存了而已
    # 开启会话看一下这两个的数值

    with tf.Session() as sess:
        # 如果不加上开启线程, 那么tf会一直运行, 等待线程启动
        # Queue_runner对象, 需要开启线程
        # 开启线程
        coord = tf.train.Coordinator()  # 构造线程协调器
        threads = tf.train.start_queue_runners(
            sess=sess,  # 需要运行哪一个会话
            coord=coord  # 传入线程协调器
        )
        # 运行
        key_sess1, value_sess1 = sess.run([key, value])

        print("key_sess1:", key_sess1)  # key_sess1: b'./dog/dog.76.jpg' # 每次是随机的, 因为我们上面勾选了, 随机读入file_queue
        print("value_sess1:", value_sess1)  # value_sess1: b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01... # 一堆字节码文件

        # 检查是否终止:
        coord.request_stop()  # 询问是否可以终止
        coord.join(threads)  # 终止线程

    # 解码 -> 将数据 转换成 tf能够识别的Tensor数据
    # 图片都是jpeg的格式, 所以我们这里使用decode_jpeg()对图片进行解码
    image_decode = tf.image.decode_jpeg(value)  # 传入一个contents -> 图片的原始文件格式
    print("image_decode:", image_decode)  # image_decode: Tensor("DecodeJpeg:0", shape=(?, ?, ?), dtype=uint8)
    # 这就是我们转化成为了Tensor对象, 但是我们可以看到shape=(?, ?, ?) 形状还没有被定义, 并且类型默认是uint8
    # 但是我们现在还不可以直接放入到批处理队列中去, 因为特征大小还不唯一, 所以我们需要统一特征

    # 统一特征处理
    # 需要将每张照片调整大小, 将每张图片都缩放到统一大小, 这样就能确定了长, 宽, 也就是shape的前两个, 但是通道数还没有确定
    image_resized = tf.image.resize_images(
        images=image_decode,  # 传入的图像
        size=[200, 200]  # 重设大小
    )
    print("image_resized:", image_resized)  # image_resized: Tensor("resize/Squeeze:0", shape=(200, 200, ?), dtype=float32)
    # 重新设置后, 我们可以看到, shape的前两项,长, 宽, 已经确定了, 但是通道数还没有确定, 所以我们在确定一下通道数, 并且现在类型已经是float32

    # 设置静态图shape, 能用静态修改, 就用静态修改, 我们知道, 我们这组图片通道数肯定为3
    image_resized.set_shape([200, 200, 3])  # 静态形状: 只有在形状没有完全固定下来的情况下, 才可以通过tensor.set_shape()进行改变, 直接对原来的张量进行更改
    print("image_resized:", image_resized)  # image_resized: Tensor("resize/Squeeze:0", shape=(200, 200, 3), dtype=float32)
    # 这样shape就已经确定完了

    # 只有形状, 类型, 都确定下来了, 才可以放到批处理队列中
    # 3. 批处理队列
    image_batch = tf.train.batch(
        tensors=[image_resized],  # 这里我认为, 返回值是高1阶的, 所以又套了个[]
        batch_size=100,  # 从队列中读取的样本个数
        num_threads=1,  # 线程数
        capacity=100  # 批处理队列的容量
    ) # -> 返回四阶张量:shape:(样本数, 每个样本的长, 每个样本的宽, 通道数)
    print(image_batch)  # Tensor("batch:0", shape=(100, 200, 200, 3), dtype=float32)

    # 开启会话:
    with tf.Session() as sess:
        # 我们仍然要开启线程
        # 构造线程协调器
        coord = tf.train.Coordinator()  # 实例化线程协调员
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 运行
        key_new, value_new, image_decode_new, image_resized_new, image_batch_new = sess.run(
            [key, value, image_decode, image_resized, image_batch])
        print("key_new:", key_new)
        print("value_new:", value_new)
        print("image_decode_new:", image_decode_new)  # 就是图片在tf中的表示形式, 就是三阶的uint类型
        print("image_resized_new:", image_resized_new)  # 重新设置大小后的, 就是缩放后的三界float32类型
        print("image_batch_new:", image_batch_new)  # 进入批处理队列后的, 是100张图片,

        # 询问是否可以终止线程
        coord.request_stop()
        coord.join(threads)  # 回收进程


if __name__ == '__main__':
    read_picture_demo()
