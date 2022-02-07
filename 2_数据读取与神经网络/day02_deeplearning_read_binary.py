# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import os


class Cifar():

    def __init__(self):
        # 设置图像大小
        self.height = 32
        self.width = 32
        self.channel = 3

        # 设置图像字节数
        self.image = self.height * self.width * self.channel
        self.label = 1
        self.sample = self.image + self.label

    def read_binary(self):
        """
        读取二进制文件
        :return:
        """
        # 1、构造文件名队列
        filename_list = os.listdir("./cifar-10-batches-bin")
        # print("filename_list:\n", filename_list)
        file_list = [os.path.join("./cifar-10-batches-bin/", i) for i in filename_list if i[-3:] == "bin"]
        # print("file_list:\n", file_list)
        file_queue = tf.train.string_input_producer(file_list)

        # 2、读取与解码
        # 2.1 读取
        reader = tf.FixedLengthRecordReader(self.sample)
        # key文件名 value样本
        key, value = reader.read(file_queue)
        print("key:", key)
        print("value:", value)

        # 2.2 解码
        image_decoded = tf.decode_raw(value, tf.uint8)
        print("image_decoded:\n", image_decoded)  # 1阶的

        # 统一特征处理:
        # 切片操作: 将目标值和特征值分开
        """
        tf.InteractiveSession() # 开启交互式Session, 这样可以直接run()或者eval()
        API: tf.slice(input_, begin, size, name=None)
        作用:
            从张量中提取想要的切片. 此操作由由begin指定位置开始的张量input中提取一个尺寸size的切片
              切片size被表示为张量形状,其中size[i]是你想要分割的input的第i维的元素的数量.切片的起始位置(begin)表示为每个input维度的偏移量.换句话说,begin[i]是你想从中分割出来的input的“第i个维度”的偏移量。
        参数:
            input_:input_类型为一个tensor，表示的是输入的tensor，也就是被切的那个
            begin: begin是一个int32或int64类型的tensor，表示的是每一个维度的起始位置, 索引值
            size: size是一个int32或int64类型的tensor，表示的是每个维度要拿的元素数, 切割的元素数(长度)
            name=None: name是指令名称，可写可不写
        return:
            返回一个和输入类型一样的tensor
        """
        label = tf.slice(image_decoded, [0], [self.label])  # 从0, 切一个
        image = tf.slice(image_decoded, [self.label], [self.image])  # 从第一个, 切image字节数的长度
        print("label:\n", label) # shape: (1, )
        print("image:\n", image) # shape: (3702, ) # 我们需要恢复tf的图像表示shape

        # 调整图像的形状, 这里使用动态形状修改
        # 因为是shape:(3702, ) -> (3, 32, 32) 实际上是跨阶了, 所以我们要使用动态形状修改, 前提: 前后元素相同!
        image_reshaped = tf.reshape(
            tensor=image, # 传入的tensor对象
            shape=[self.channel, self.height, self.width] # 需要改变的形状
        )
        print("image_reshaped:\n", image_reshaped) # shape(3, 32, 32)
        # 但是这个shape不符合我们tf中对图像要求的shape(长, 宽, 通道数)
        # 所以我们需要进行转至处理
        # 三维数组的转置: 将图片的shape转换成为tf中默认的
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0]) # 表示新的对应的位置索引
        print("image_transposed:\n", image_transposed) # shape(32, 32, 3) dtype=uint8, 形状转换过来了

        # 调整图像类型
        tf.cast(image_transposed, dtype=tf.float32) # 这样就可以进行批处理了

        # 3、构造批处理队列
        label_batch, image_batch = tf.train.batch(
            [label, image_transposed],
            batch_size=100, # 一次性处理100个
            num_threads=2, # 线程数
            capacity=100 # 容量
        )
        print("label_batch:", label_batch) # shape(100, 1) 标签对象
        print("image_batch:", image_batch) # shape(100, 32, 32, 3)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()  # 线程协调器
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            label_value, image_value, image_decoded_value, label_new, value_new, image_reshaped_new, label_batch_new, image_batch_new = sess.run(
                [label_batch, image_batch, image_decoded, label, image, image_reshaped, label_batch, image_batch])
            print("label_value:\n", label_value)  # 文件名
            print("image:\n", image_value)  # 内容
            print("image_decoded_value:", image_decoded_value)  # 这个就是解码之后的数据
            # 也就是在tensor中的表示形式, 也就是一个一维数组, 第一个表示类别, 后面表示图片信息
            print("label_new:", label_new) # [1] -> 目标值的值
            print("value_new:", value_new) # 剩余的图片数字值
            print("image_reshaped_new:", image_reshaped_new) # reshape后的, 对象
            print("label_batch_new:", label_batch_new)  # 标签值
            print("image_batch_new:", image_batch_new)

            coord.request_stop()
            coord.join(threads)

        return image_value, label_value

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将样本的特征值和目标值一起写入tfrecords文件
        :param image:
        :param label:
        :return:
        """
        with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                # print("tfrecords_image:\n", image)
                # print("tfrecords_label:\n", label)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))
                # example.SerializeToString()
                # 将序列化后的example写入文件
                writer.write(example.SerializeToString())

        return None

    def read_tfrecords(self):
        """
        读取TFRecords文件
        :return:
        """
        # 1、构造文件名队列
        file_queue = tf.train.string_input_producer(["cifar10.tfrecords"])

        # 2、读取与解码
        # 读取
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # 解析example
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = feature["image"]
        label = feature["label"]
        print("read_tf_image:\n", image)
        print("read_tf_label:\n", label)

        # 解码
        image_decoded = tf.decode_raw(image, tf.uint8)
        print("image_decoded:\n", image_decoded)
        # 图像形状调整
        image_reshaped = tf.reshape(image_decoded, [self.height, self.width, self.channel])
        print("image_reshaped:\n", image_reshaped)

        # 3、构造批处理队列
        image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size=100, num_threads=2, capacity=100)
        print("image_batch:\n", image_batch)
        print("label_batch:\n", label_batch)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            image_value, label_value = sess.run([image_batch, label_batch])
            print("image_value:\n", image_value)
            print("label_value:\n", label_value)

            # 回收资源+
            coord.request_stop()
            coord.join(threads)

        return None


if __name__ == "__main__":
    cifar = Cifar()
    # image_value, label_value = cifar.read_binary()
    # cifar.write_to_tfrecords(image_value, label_value)
    cifar.read_tfrecords()
