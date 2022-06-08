import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

import os


def read_picture():
    """
    读取狗图片案例
    :return:
    """
    # 1、构造文件名队列
    # 构造文件名列表
    filename_list = os.listdir("./dog")
    # 给文件名加上路径
    file_list = [os.path.join("./dog/", i) for i in filename_list]
    # print("file_list:\n", file_list)
    # print("filename_list:\n", filename_list)

    file_queue = tf.train.string_input_producer(file_list)  # 这是一个文件队列, 我们需要开启线程
    # string_tensor, 需要的是一阶张量, 但是我们说过, 只需要传入文件列表就可以, 底层会默认转换成列表

    # 2、读取与解码
    # 读取
    reader = tf.WholeFileReader()  # 返回一个读取器
    key, value = reader.read(file_queue)  # 使用read(文件队列)方法,
    print("key:\n", key)  # key是文件名
    print("value:\n", value)  # value是一张图片的原始编码形式
    with tf.Session() as sess:
        # 别忘了开启线程
        key_sess, value_sess = sess.run([key, value])
        print("key_sess:", key_sess)
        print("value_sess:", value_sess)

    # 解码
    # 这里使用的是jpeg的格式, 所以我们使用decode_jpeg()
    image_decoded = tf.image.decode_jpeg(value)
    print("image_decoded:\n", image_decoded)  # -> Tensor对象, shape(?, ?, ?), 解码默认为uint8
    # image_decode就是解码后的Tensor对象, 也是tf中的uint8类型的数据.
    # 但是我们现在还不能直接放到批处理队列中, 因为图片的特征大小还不一样, 所以我们需要统一特征
    # 必须要确定(处理) shape
    # 若直接处理, 则会报错

    # 将图片缩放到同一个大小
    image_resized = tf.image.resize_images(image_decoded, [200, 200])
    # 参数分别是img, 和size
    print("image_resized_before:\n", image_resized)  # (但是通道数还没有确定)
    # 更新静态形状 直接set_shape
    # 有一个规则, 是能用静态形状就用静态形状
    image_resized.set_shape([200, 200, 3])
    print("image_resized_after:\n", image_resized)

    # 只有形状, 都确定下来了, 才可以放入到批处理队列
    # 3、批处理队列
    image_batch = tf.train.batch([image_resized], batch_size=100, num_threads=2, capacity=100)
    print("image_batch:\n", image_batch)  #
    # 此时shape=[100, 200, 200, 3] 就是四阶张量, 是100张图片值

    # 开启会话
    with tf.Session() as sess:
        # 开启线程
        # 构造线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 运行 -> 查看一下张量的值
        key_new, value_new, image_decoded_new, image_resized_new, image_batch_new = sess.run(
            [key, value, image_decoded, image_resized, image_batch])
        print("key_new:\n", key_new)
        print("value_new:\n", value_new)
        print("image_decoded_new:", image_decoded_new)
        print("image_resized_new:\n", image_resized_new)
        print("image_batch_new:\n", image_batch_new)

        coord.request_stop()  # 请求(询问)停止
        coord.join(threads)  # 请求回收线程

    return None


if __name__ == "__main__":
    # 代码1：读取狗图片案例
    read_picture()
