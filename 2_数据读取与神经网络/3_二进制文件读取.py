"""




"""

import tensorflow._api.v2.compat.v1 as tf
import os

tf.disable_v2_behavior()


# 我们这里使用面向对象的方式
class Cifer(object):
    def __init__(self):
        # 初始化操作
        # 图片的形状
        self.height = 32
        self.width = 32
        self.channels = 3

        # 字节数
        self.image_bytes = self.height * self.width * self.channels # 特征的字节数
        self.label_bytes = 1 # 标签字节数
        self.all_bytes = self.image_bytes + self.label_bytes # 总的字节数, 也就是一个样本=总共的字节数

    # 二进制文件读取API:
    def read_and_decode(self, file_list):
        # 1. 构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)
        # 2. 读取与解码, 以及数据处理
        # 2.1 读取
        reader = tf.FixedLengthRecordReader(self.all_bytes) # 以二进制方式读取数据, 需要指定多少个字节为一个样本
        key, value = reader.read(file_queue) # 传入文件名队列, read会随机抽取一个文件, 读取进来
        # 其中key是文件名, value就是样本值
        # print("key: ", key) # Tensor("ReaderReadV2:0", shape=(), dtype=string)
        # print("value:", value) # value: Tensor("ReaderReadV2:1", shape=(), dtype=string)

        """
        ### 这个地方没有看懂 ###
        # 解析example
        feature = tf.parse_single_example(
            value,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)
            }
        )
        image = feature["image"]
        label = feature["label"]
        print("read_tf_image:\n", image) #  Tensor("ParseSingleExample/ParseExample/ParseExampleV2:0", shape=(), dtype=string)
        print("read_tf_label:\n", label) #  Tensor("ParseSingleExample/ParseExample/ParseExampleV2:1", shape=(), dtype=int64)
        """

        # 解码:
        value_decode = tf.decode_raw(value, tf.uint8) # 解析的样本 和类型
        print("value_decode", value_decode) # image_decode Tensor("DecodeRaw:0", shape=(?,), dtype=uint8)

        # 切分目标值和特征值
        # API: tf.slice() # 从输入, 开始索引,切size大小
        label = tf.slice(
            input_=value_decode, # 要切割的对象
            begin=[0], # 从哪个索引出开始切割
            size=[self.label_bytes] # 切割长度
        )
        image = tf.slice(input_=value_decode, begin=[self.label_bytes], size=[self.image_bytes])
        print("label: ", label) # 标签 label:  Tensor("Slice:0", shape=(1,), dtype=uint8)
        print("image: ", image) # 图片 image:  Tensor("Slice_1:0", shape=(3072,), dtype=uint8)


        # 图像形状调整: 因为原来是1阶, 我们要转换成3阶, 跨阶转换形状
        # 所以我们需要使用动态形状转换, 条件: 转换前后的元素个数一致
        image_reshape = tf.reshape(value_decode, shape=[self.channels, self.height, self.width]) #
        print("image_reshape: ", image_reshape) # Tensor("Reshape:0", shape=(3, 32, 32), dtype=uint8)
        # 已经转换成shape:[channels, height, width] # 但是不符合TF中对图片形状
        # 调整成TF中默认的图片形状: [height, width, channels]
        # 于是我们对image_reshape进行转置处理:
        image_transhosed = tf.transpose(
            image_reshape, # 需要转至的Tensor变量
            perm=[1, 2, 0] # 按照perm对输入进行转置
            # 是将 0 1 2 -> 1 2 0 上的位置上去.
        )
        print("image_transposed:", image_transhosed) # Tensor("transpose:0", shape=(32, 32, 3), dtype=uint8)
        # 此时shape就已经是(32, 32, 3) -> 此时就满足TF默认对图片的基本形状[height, width, channels]
        # 此时, 形状, 类型, 已经确定下来了, 此时我们就可以放入批处理队列当中去了


        # 3. 批处理
        # 构造批处理队列:
        label_batch, image_batch = tf.train.batch(
            tensors=[label, image_transhosed], # 需要放入到批处理的队列. 注意, 一定是4阶
            batch_size=100,
            num_threads=2, # 线程数
            capacity=100 # 容量
        )
        print("label_batch:", label_batch) # label_batch: Tensor("batch:0", shape=(100, 1), dtype=uint8)
        print("image_batch:", image_batch) # image_batch: Tensor("batch:1", shape=(100, 32, 32, 3), dtype=uint8)


        # 开启会话:
        with tf.Session() as sess:
            # 开启线程:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) #

            # 接下来我们可以看一下每一部分是怎么变化的, 他们的值是如何变化的
            sess.run([label, image, ])

            coord.request_stop()
            coord.join(threads)
        return None



if __name__ == '__main__':
    file_name = os.listdir("./cifar-10-batches-bin")  # 获取文件夹中的文件列表
    # print(file_name)
    # 使用列表生成式, 直接生成file_list, 我们只需要.bin结尾的
    file_list = [os.path.join("./cifar-10-batches-bin", file) for file in file_name if file[-3:] == "bin"]
    # 实例化对象
    cifer = Cifer()
    cifer.read_and_decode(file_list=file_list) # 调用读取文件api
