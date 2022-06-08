"""
深度学习 day02

数据IO操作
    tf中有三种：
    1. 预加载数据: 最简单的: 直接将数据保存到一个变量中: data = ...我们基本不会使用
    2. Feeding: 占位符 & run(feed_dict)搭配使用, 运行每一步时, python代码提供数据
    3. QueueRunner的方式. -> 基于队列的输入管道从TensorFlow图形开头的文件中读取数据
        通用文件读取流程: 不同的文件, 方法也是不一样的.
            图片
            二进制数据
            TFRecords -> tf的专门的数据文件
神经网络基础
    神经网络原理
    手写数字识别案例

3.1 通用的文件读取流程:
    多任务: 一个进程读取数据, 一个进程训练数据, 然后将数据线程传输的数据传输给训练进程
    这样我们的效率才能最大化。
    也就是 -> 多线程 + 队列 的这种模式
    3.1.1 通用的文件读取流程
        三个阶段： 构造文件名队列, 读取与解码, 批处理
        大体流程:
        构造文件名队列: 我们需要将文件名字随机(random)加载进入我们的: Filename Queue, 然后抽取一个样本, 去读取数据, 针对不同文件这个样本的概念
            不同, 比如文本文件, 一个样本就是一行, 图片样本就是这个图片, 二进制文件, 可能就是一个固定字节的二进制码
        读取与解码: 然后我们需要解码, 不同类型的文件, 编码不同, 所以我们需要解码成为数据(tensor对象), 得到解码之后的一个样本.
        批处理阶段: 有一个Example Queue, 通过前面不断的生成样本, 放入到这个队列中, 然后与另一个训练进程进行通讯.
        手动开启进程: 注意这些操作需要启动这些队列操作的线程, 以便我们能够在进行文件读的过程中能够顺利进入入队和出队操作
        最终 -> 这就让我们的数据读取和训练模型变得比较高效了起来

        details: 
        1）构造文件名队列： 需要将读取的文件名放入到文件名队列
            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True, num_epochs=None)
                参数:
                    string_tensor: 含有文件名 + 路径 的列表 (含有文件名+路径的1阶张量 (底层: 列表 -> 张量))
                    shuffle: random shuffle, 是否打乱文件名读取的顺序, 默认为: True
                    num_epochs: 过几遍数据, 默认无限过数据, 文件数量比较少, 但是批处理队列比较大, 那么他就会重复使用以前的文件, 所以这里可以限定读几遍
                return:
                    文件队列对象: file_queue

        2）读取与解码
            从队列当中读取文件内容, 并进行解码操作
            读取: TensorFlow默认每次只读取一个样本，具体到文本文件读取一行、二进制文件读取指定字节数(最好一个样本)、图片文件默认读取一张图片、TFRecords默认读取一个example
            解码: 对于读取不同的文件类型，内容需要解码操作，解码成统一的Tensor格式
        读取文件内容:
            阅读器默认每次只读取一个样本,具体到文本文件读取一行、二进制文件读取指定字节数(最好一个样本)、图片文件默认读取一张图片、TFRecords默认读取一个example
            文本：
                读取：tf.TextLineReader()
                    阅读文本文件逗号分隔值（CSV）格式,  默认按行读取
                    return：读取器实例(对象)
                解码: tf.decode_csv(records, record_defaults=None, field_delim=None, name=None)：解码文本文件内容
                    将csv文件转换为张量，与tf.TextLineReader搭配使用
                    records：tensor型字符串，每个字符串是csv中的记录行
                    field_delim：默认分隔符“,”
                    record_defaults：参数决定了所得张量的类型，并设置一个值。例如：[[1], ["None"]] 表示第一列为int，第二列为字符串
            图片：
                读取：tf.WholeFileReader()
                    用于读取图片文件。
                    return：读取器实例
                解码：
                    tf.image.decode_jpeg(contents)
                        将JPEG编码的图像解码为uint8张量
                        return:uint8张量，3-D形状[height, width, channels]
                            (channels是通道数)
                    tf.image.decode_png(contents)
                        将PNG编码的图像解码为uint8张量或unit16张量
                        return:张量类型，3-D形状[height, width, channels]
            二进制：
                读取：tf.FixedLengthRecordReader(record_bytes)
                    要读取每个记录是固定数量字节的二进制文件
                    record_bytes:整型，指定每次读取(一个样本)的字节数
                    return：读取器实例
                解码：tf.decode_raw()
                    将字节转换为一个数字向量表示，字节为一字符串类型的张量
                    与tf.FixedLengthRecordReader搭配使用，二进制读取为uint8格式
            TFRecords -> tf中专门设计的一种文件格式
                读取：tf.TFRecordReader()
                    读取TFRecords文件
            最后:
        1、他们有共同的读取方法：read(file_queue)：从队列中指定数量内容返回一个Tensors元组（key文件名字，value默认的内容(一个样本)）
            reader.read(文件队列) -> return: (key, value), key是文件名字, value是文件的内容
                key, value = 读取器.read(file_queue)
                返回一个tensor元组:
                key：文件名
                value：一个样本(默认值)
        2、由于默认只会读取一个样本，所以通常想要进行批处理。使用tf.train.batch或tf.train.shuffle_batch进行多样本获取，便于训练时候指定每批次多个样本的训练
            需要用到batch或者suffle_batch将一个样本加载进我们的批处理队列中去.


        解码阶段. 默认所有的内容都解码成tf.uint8类型, 如果只有需要转换成指定类型则可以使用tf.cast()

        3）批处理队列:
            在解码之后，我们可以直接获取默认的一个样本内容了，但是如果想要获取多个样本，
                这个时候需要结合管道的末尾进行批处理
            tf.train.batch(tensors, batch_size, num_threads = 1, capacity = 32, name=None)
                作用: 读取指定大小（个数）的张量
                参数:
                    tensors：可以是包含张量的列表,  批处理的内容放到列表当中
                    batch_size:  从队列中读取的批处理大小, 一次性要处理几个样本
                    num_threads：进入队列的线程数 -> 看CPU核心数目
                    capacity：整数，队列中元素的最大数量, 队列的容量.
                return:
                    tensors: 返回批处理后的tensor对象了. 以一个列表返回
            tf.train.shuffle_batch()

        手动开启线程:
            以上用到的队列都是tf.train.QueueRunner对象.
            每个QueueRunner都负责一个阶段,在会话中开启: tf.train.start_queue_runners函数会要求图中的每一个
                QueueRunner启动他的运行队列操作的线程. (这些操作需要在会话中开启)

            tf.train.QueueRunner()

            开启会话后, 在session中：
                tf.train.start_queue_runners(sess=None, coord=None)
                    作用: 收集图中所有的队列线程，默认同时启动线程
                    参数:
                        sess：所在的会话
                        coord：线程协调器
                    return：
                        返回所有线程

                tf.train.Coordinator()
                    作用: 线程协调员，对线程进行管理和协调
                    return：
                        线程协调员实例(obj)
                    方法: obj.
                        request_stop()：请求停止
                        should_stop()：询问是否结束
                        join(threads=None, stop_grace_period_secs=120)：回收线程

                实例化线程协调器, 然后传给start_queue_runners(), 询问线程是否执行完毕, 然后关闭线程.

3.2 图片数据
    3.2.1 图像基本知识
    如何将图片文件转换成机器学习算法能够处理的数据?
        文本  特征词 -> 二维数组 -> shape(n_samples, m_features)
        字典  one-hot -> 二维数组 -> shape(n_samples, m_features)
        图片  像素值 -> 三维数组 -> shape(图片长度, 图片宽度, 图片通道数)
        1 图片三要素
            组成一张图片的特征值是所有的像素值. 有三个维度: 图片长度, 图片宽度, 图片通道数.
            黑白图 也称 灰度图 shape:(长, 宽, 1)
                一个通道
                    黑 [0, 255] 白
            彩色图 shape:(长, 宽, 3)
                三个通道
                    一个像素点 是三个通道值构成, 每个通道又是一个灰度图
                    R [0, 255]
                    G [0, 255]
                    B [0, 255]
            假设一张彩色图片的长200, 宽200, 通道数为3, 那么总的像素数量为200*200*3, 而像素点为200*200

        2 TensorFlow中表示图片: 张量形状
            Tensor对象: Tensor(指令名称, shape, dtype)
            在TensorFlow中如何用张量表示一张图片呢?
            一张图片就是一个3D张量，[height, width, channel]，height就表示高，width表示宽，channel表示通道数。我们会经常遇到3D和4D的表示
            Tensor对象
                指令名称、形状、类型
                shape = [height, width, channel]
            所以:
            单个图片：[height, width, channel]
            多个图片（4D）：[batch, height, width, channel]，batch表示批数量


        3 图片特征值处理
            iris数据集: 150个样本, 4个特征, 还有对应的目标值
            等等...
            这些数据集是比较小的, 样本数目比较少, 特征也比较少,
            而一张图片就是一个样本, 里面所包含的像素个数就是特征值, 所以对于一张图片,他的特征值是非常庞大的
            并且对数据进行批量的操作, 如果每一张图片不一样大小, 那么在批量操作时候就会发生阻碍.
            综上: 为什么我们需要图片特征值处理
                - 样本特征 数据量 大
                - 样本和样本形状不统一, 所以无法进行批量操作和运算

        为什么要缩放图片到统一大小？
            在进行图片识别的时候，每个图片样本的特征数量要保持相同（方便神经网络的训练）。
            所以需要将所有图片张量大小统一转换。另一方面如果图片的像素量太大，
            也可以通过这种方式适当减少像素的数量，减少训练的计算开销
            综上:
                1) 每一个样本特征数量要一样多
                2) 缩小图片的大小
            缩放处理:
            tf.image.resize_images(images, size)
                作用: 缩小放大图片
                参数:
                    images：4-D形状[batch, height, width, channels]，或3-D形状的张量[height, width, channels]的图片数据
                    size：1-D int32张量：(new_height, new_width)，图像的新尺寸
                返回:
                    传进来几阶, 返回的就是几阶, 4-D格式或者3-D格式图片, 返回缩放后的图片
        4 数据格式
            存储：uint8 -> 节约空间
            训练：float32 -> 提高精度

    3.2.4 案例：狗图片读取
        1）构造文件名队列
            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
        2）读取与解码
            读取：
                reader = tf.WholeFileReader()
                key, value = reader.read(file_queue)
            解码：
                image_decoded = tf.image.decode_jpeg(value)
            并且我们还需要使样本的形状和类型统一
                resize_images()
        3）批处理队列
            image_decoded = tf.train.batch([image_decoded], 100, num_threads = 2, capacity=100)
        手动开启线程

3.3 二进制数据:
    数据集信息:
        CIFAR10二进制数据集介绍.
        每3073字节是一个样本
        1个目标值 + 3072像素
        第一个字节是图像的标签(目标值), 他是0~9的一个数字, 接下来的3072个字节是图像
        像素的值, 前1024是红色通道值, 接着1024是绿色, 最后是蓝色.

    tensor对象
        shape:[height, width, channel] -> [32, 32, 3] [0, 1, 2] -> []
        [[32 * 32的二维数组],
        [32 * 32的二维数组],
        [32 * 32的二维数组]]
            --> [3, 32, 32] [channel, height, width] 三维数组的转置 [0, 1, 2] -> [1, 2, 0]
            [3, 2] -转置-> [2, 3]
        1）NHWC与NCHW
        T = transpose 转置
        CIFAR10二进制数据集介绍
            每3073个字节是一个样本
            1个目标值 + 3072像素(1024字节红色通道, 1024字节绿色通道, 1024字节蓝色通道

    3.3.2 CIFAR10 二进制数据读取 流程分析:
        流程分析：
            1）构造文件名队列
                file_queue = tf.train.spring...(文件名队列)
            2）读取与解码
                # 读取
                    reader = tf.FixeddLengthRecordReader(3073) # 以3072个字节为一个样本
                    key, value = reader.read(file_queue)
                # 解码
                    decoded = tf.decode_raw(value, tf.uint8)
                一个样本包含了目标值和特征值, 所以我们需要进行分割
                # 将数据的标签和图片进行分割, 使用切片进行分割
                    对Tensor对象进行切片
                        tf.slice()进行切片
                    label: 目标值
                    feature: 一个样本image(3072字节 = 1024r+1024g+1024b)
                        [[1024], -> [   [r[32, 32]],
                        [1024],         [g[32, 32]],
                        [1024]]         [b[32, 32]]    ]
                    (每个图片都是, 32 * 32的(大小),3个通道, 然后我们吧图片拍扁了, 所以得到了1024+1024+1024,分别是32个像素*32个像素的通道值
                    所以形状: shape=(3, 32, 32) = (channels, height, width)
                    需要转换成 => TensorFlow的图像表示习惯, shape(height, width, channels)
                        ndarray.T 转至, 行变成列, 列变成行
                        (3, 32, 32) -> (32, 32, 3)
                    这里的图片形状设置从1维到3维数据的时候, 涉及到NHWC和NCHW的概念:
                    在读取设置图片形状的时候有两种格式:
                    设置为: "NHWC"时: 排列顺序为: [batch, height, width, channels] -> tf默认的图像形式
                    设置为: "NCHW"时: 排列顺序为: [batch, channels, height, width]
                    N表示这批图像有几张, H表示图像在竖直方向有多少像素, W表示水平方向像素数, C表示通道数

                    所以我们针对上面的shape, 需要进行转至:
                        位置索引: 3, 32, 32 对应的位置索引: 0, 1 ,2
                        需要变成: 32, 32, 3 对应的索引; 1, 2 ,0
                    API:
                        tf.transpose(a, perm=None, name="transpose", conjugate=False):
                            置换 a,根据 perm 重新排列尺寸.
                        参数:
                            a: 需要转至的tensor对象
                            perm: 转换后的位置索引
                切片, 形状处理后, 我们就可以进入批处理队列中去了:
            3）批处理队列
            开启会话
            手动开启线程

    案例: 二进制文件读取
3.4 TFRecords
    3.4.1 什么是TFRecords文件
    3.4.2 Example结构解析
        cifar10
            特征值 - image - 3072个字节
            目标值 - label - 1个字节
        example = tf.train.Example(features=tf.train.Features(feature={
        "image":tf.train.Feature(bytes_list=tf.train. BytesList(value=[image])
        "label":tf.train.Feature(int64_list=tf.train. Int64List(value=[label]))
        }))
        example.SerializeToString()
    3.4.3 案例：CIFAR10数据存入TFRecords文件
        流程分析
    3.4.4 读取TFRecords文件API
        1）构造文件名队列
        2）读取和解码
            读取
            解析example
            feature = tf.parse_single_example(value, features={
            "image":tf.FixedLenFeature([], tf.string),
            "label":tf.FixedLenFeature([], tf.int64)
            })
            image = feature["image"]
            label = feature["label"]
            解码
            tf.decode_raw()
        3）构造批处理队列
3.5 神经网络基础
    3.5.1 人工神经网络， 简写ANN。
        由： 输出层 隐藏层 输出层 三层结构组成
        感知机 <==> 逻辑回归 很相似
        只不过感知机是使用的sign函数, 而逻辑回归使用的是sigmoid
        使用playground网站, 可视化神经网络

        感知机可以解决的问题
        输入层
            特征值和权重 线性加权
            y = w1x1 + w2x2 + …… + wnxn + b
            细胞核-激活函数
                sigmoid
                sign
        隐藏层
        输出层

    单个神经元 -> 感知机
    感知机(PLA: Perceptron Learning Algorithm))
        感知机可以解决的问题 -> 简单的或, 与问题 (线性二分类问题)
        或: x1 x2
            0,0 -> 0
            0,1 -> 1
            1,0 -> 1
            1,1 -> 1
        与:
            0,0 -> 0
            1,0 -> 0
            0,1 -> 0
            1,1 -> 1
        异或: 此时一个神经元就已经无法区分了, 和线性激活函数就已经无法分辨了
            0,0 -> 0
            0,1 -> 1
            1,0 -> 1
            1,1 -> 0
        # 我们就需要添加神经元, 或者改变网络结构, 或者换不同的激活函数



        单个神经元不能解决一些复杂问题
        1) 多层神经元
        2) 改变神经网络结构
        2) 增加激活函数

3.6 神经网络原理
    神经网络的主要用途就是在于分类, 那么整个神经网络的原理是怎么样的我们还是围绕着损失, 优化这两方面去说/
    并且神经网络的输出结果如何分类?
        回归 -> sigmoid函数映射 -> 实现二分类问题
        回归 -> softmax回归 -> 实现多分类问题
    如何调整神经网络:
    思路: 构造损失函数, 优化损失

    全连接神经网络如何实现多分类问题:
        softmax回归:
            softmax(yi) = e^yi / (sum(e^yi))
        logits + softmax映射 -> 就可以解决多分类问题
        如何找到比较准确的模型参数呢, 就是构造损失函数. 并且不断优化它
    如何构造 全连接神经网络 损失函数:
        回顾: 线性回归:
            损失函数: 均方误差
        回顾: 逻辑回归:
            损失函数: 对数似然函数
            y = w1x1 + w2x2 + …… + wnxn + b

    在神经网络中, 我们如何优化参数呢, 依靠的是损失函数
    那么神经网络中的损失函数: 交叉熵损失
    公式:
        H_y^'(y) = -sum_i(y_i^'log(y_i))
        hy'(y) = -sum(y'i log (yi))
        过程:
            预测的类别用one-hot编码, 然后先用回归函数得到函数值, 经过softmax映射
            得到预测值, 然后 -sum(预测概率 * 类别(0, 1的值))
            所以我们当预测概率值最大*1 表示真实的类别, 此时前面再加一个-, 所以此时损失函数就是最小, 也是我们预测对的时候.
        总损失: 然后计算每一个样本的损失值, 求和,或者求取平均, 就是我们对于这个模型衡量的整个损失函数.
    优化损失:
        训练过程中计算机会尝试一点点增大或者减小每个参数, 看如何能够减小相比于训练数据集的误差,以找到最优的权重, 偏置参数组合
        比如我们也可以使用梯度优化算法 去优化我们的损失

    API:
        softmax和交叉熵损失结合的API:
            tf.softmax_cross_entropy_with_logits(labels=None, logits=None, name=None)
                作用: 计算logits和labels之间的交叉损失熵
                参数:
                    labels: 标签值(真实值) -> one-hot编码之后的值.
                    logits: 样本加权之后的值, 回归的输出值
                    name: 指令名称
                return:
                    返回损失值列表
            tf.reduce_mean(input_tensor)
                作用: 计算张量的尺寸的元素平均值
        这样我们就可以用一个参数来表示我们的损失.
                
3.7 案例：Mnist手写数字识别
    3.7.1 数据集介绍
    数据集分为两部分: 55000行的训练数据集(mnist.train), 和10000行的测试数据集(mnist.test).
    每个数据单元有两部分组成: 一张包含手写数字的图片和一个对应的标签,
    我们把这些图片设为xs, 把这些标签设为ys, 训练集和测试集都包含xs和ys, 比如: 训练集的图片是mnist.train.image, 训练集的标签是mnist.train.label
    特征值:
        这些图片都是黑白图片, 每一张图片包含28 * 28像素,
        我们把这个数组展开成为一个向量, 长度就是784个元素
    目标值:
        分类: one-hot编码:
            0 1 2 ... 9
            0 0 1 ... 0
            那一列标记为1, 就表示真实值.
    如何去用这个数据集:
        TF自带了这个数据集的接口, 所以不需要自行读取:
        from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets(path, one_hot=True)
            返回的对象:
            image, labels = mnist.train.next_batch() 提供批量获取的功能
            mnist.train.image 或者 labels
            mnist.test.image 或者 labels
    全连接层计算:
        就是矩阵相乘:
        y = w1x1 + w2x2 + ... +b
        x[None, 784] * w[784, 10] + bias[10] = y[None, 10] 就是我们的预测值
        None表示样本数量, 也就是多少张图片, y中的10, 需要表示one-hot编码类别. ->  所以w就是[784, 10]
        然后在使用softmax: tf.mm.softmax_cross..(labels=y_true, logitis=y_pred) 就可以使用softmax映射了
        返回的就是损失值列表, 然后我们求取平均, 然后我们在优化损失, 使用梯度下降方法,优化损失
        API:
        tf.matmul(a, b, name=None) + bias
            return: 返回全连接结果, 供交叉损失运算
        tf.train.GradienDe(learning_rate)
            梯度下降算法
            learning_rate: 学习率
            method:
                minimize(loss): 最小优化损失
        准确率计算:
            1. 比较输出的结果最大值所在位置和真实值的最大值所在位置
                np.argmax() 返回最大值所在的位置
                tf.argmax(a, axis)
                    按照axis求最大值的位置.
                tf.argmax(y_true, 1)
                tf.argmax(y_predict, 1)
                然后查看是否一致:
                tf.equal() # 如果一致, 则返回True, 如果不一致则返回Flase
                然后转换成0,1
                tf.cast(equal, tf.float32)
                # 然后求平均:
                accuracy = tf.reduce_mean()

        1 特征值
            [None, 784] * W[784, 10] + bias[10] = [None, 10] -> y_predict
            构建全连接层：
            y_predict = tf.matmul(x, W) + Bias
            构造损失：
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict,name=None)
            error = tf.reduce_mean(loss) # 就是平均损失

            如何计算准确率?
            np.argmax(y_predict, axis=1)
            tf.argmax(y_true, axis=1)
                y_predict [None, 10]
                y_true [None, 10]
            tf.equal()
            如何提高准确率？
                1）增加训练次数
                2）调节学习率
                3）调节权重系数的初始化值
                4）改变优化器
"""