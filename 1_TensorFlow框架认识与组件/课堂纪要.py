"""
深度学习 3天
TensorFlow框架的使用 1天
IO数据读取、神经网络基础 1天
卷积神经网络 -> 案例: 验证码识别 1天

今天学习的内容： 介绍TensorFlow
1、深度学习介绍
2、TensorFlow框架的使用
    1）TensorFlow的结构(整体)
    2）TensorFlow的各个组件
        图 -> graph
        会话 -> session
        张量 -> Tensor
        变量 -> Variable
    3）简单的线性回归案例 - 将TensorFlow用起来
        从头开始实现一下 -> 使用TensorFlow


1.1 深度学习与机器学习的区别
    三个方面:
    1.1.1 特征提取方面
        机器学习的特征工程需要手动完成, 并且需要领域内的专业知识.
        深度学习,通常由多个层组成, 他们通常将更简单的模型组合在一起, 将数据从一层传递到另一层来构建复杂的社会.
            通过训练大量数据自动得出模型的参数, 而不需要人工特征提取环节.

    1.1.2 数据量和计算性能要求
        深度学习需要大量的训练数据集
        训练神经网络需要大量算力.计算缓慢

    1.1.3 算法代表
         机器学习: 传统的统计学算法, 例如朴素贝叶斯, 决策树等.
         深度学习: 人工神经网络

1.2 深度学习的应用场景
     图像处理
     自然语言处理(NLP)
     语音技术

1.3 深度学习框架介绍
    1.3.1 常见深度学习框架对比
        TensorFlow, caffe, caffe2, PyTorch
    1.3.2 TensorFlow的特点
        高度灵活
        语言多样
        设备支持
        Tensorboard可视化

    1.3.3 TensorFlow的安装
        1 CPU版本
        2 GPU版本
        对比:
            CPU:诸葛亮
                综合能力比较强, 计算, 资源调度
                核芯的数量更少
                更适用于处理连续性（sequential）任务。
            GPU:臭皮匠
                专做某一个事情很好 -> 计算
                核芯的数量更多
                更适用于并行（parallel）任务

        矩阵运算: GPU速度会更快, 但是GPU需要配置CUDA

    2.0版本后是在compat.v1.Session()开启会话

2.1 TF数据流图
    2.1.1 案例：TensorFlow实现一个加法运算
    通过上述案例, 我们对TensorFlow进行分析
    2 TensorFlow结构分析
        一个构建图阶段 -> 定义了数据和操作的一个步骤
            流程图: 定义数据（张量Tensor）和操作（节点Operation）
        一个执行图阶段 -> 运行起来, 使用会话执行构建好的图中的操作
            开启会话: 调用各方资源，将定义好的数据和操作运行起来

        - 图和会话:
            图: 这是TensorFlow将计算表示为指令之间的依赖关系的一种表示
            会话: 这是TensorFlow跨一个或者多个本地或远程设备运行数据流图的机制
        - 张量: TensorFlow中的基本数据对象, 本质上是一个可供GPU加速运算的一个矩阵
        - 节点: 提供图当中执行的操作, Operation

        想要执行这些操作, 那么就需要一个绘画(针对静态图)

    2.1.2 数据流图介绍
        TensorFlow:
        Tensor - 张量 - 数据
        Flow - 流动

    TensorFlow是一个采用数据流图(data flow graphs)用于数值计算的开源框架
    节点(operation)在图中表示数学操作, 线(edges)则表示在节点间相互联系的多维数组, 即张量(tensor)

2.2 图与TensorBoard
    2.2.1 什么是图结构
        图包含了一组tf.Operation代表的计算单元对象和tf.Tensor代表的计算单元之间流动的数据
        简单来说: 图结构:
            数据（Tensor） + 操作（Operation）

    2.2.2 图相关操作:
        1 默认图, 通常tf会默认帮助我们创建一个图
            查看默认图的方法:
                1）调用方法
                    通过调用tf.get_default_graph()访问, 要将操作添加到默认图形中, 直接创建OP即可
                2）查看属性
                    .graph
                    op和session的属性, 默认是同一张图

        2 创建图, 自定义图
            API:
                tf.Graph()
                返回一个自定义图对象
            如果要在这张图中创建OP, 典型的用法是使用tf.Graph.as_default()上下文管理器
                在 上下文管理器 中定义数据和操作, 这样我们的数据和操作就在我们自己的图当中.
            例如:
                new_g = tf_Graph()
                with new_g.as_default():
                    定义数据和操作
                with tf.Session(graph=new_g) as new_sess:
                    开启绘会话, 需要Session指定graph
                    此时的new_sess就属于new_g中

    TensorFlow有一个亮点, 就是通过TensorBoard进行可视化

    2.2.3 TensorBoard:可视化学习
    实现程序可视化的步骤:
        1 数据序列化 -> events文件, 将图序列化到本地
            tf.summary.FileWriter(path, graph)
                返回FilterWriter,将图对象写入(序列化)事件文件到指定目录
                path: 需要保存的路径
                graph: 哪一张图需要可视化.
        2 tensorboard
            通过cmd, 启动tensorboard
                tensorboard --logdir="path"
            在浏览器中打开TensorBoard的图页面,127.0.0.1:6006
            然后就可以看到可视化的效果了.
        可视化中心:
            左边是图例, 中心就是我们的数据流图.

    2.4 OP
        数据：Tensor对象
        操作：Operation对象(节点) - Op
        1 常见OP
                类型              实例
            标量运算
            向量运算
            矩阵运算
            带状态的运算
            神经网络组件
            存储, 恢复
            队列及同步运算
            控制流

        区分:
            操作函数                        &                  操作对象
            tf.constant(Tensor对象)            输入Tensor对象(我们的数据) -> 创建Const对象 -> 输出 Tensor对象
            tf.add(Tensor对象1, Tensor对象2)   输入Tensor对象1, Tensor对象2 -> 经过Add对象 -> 输出 Tensor对象3
            就是: 输入Tensor对象, 然后经过操作函数, 操作函数会实例化一个操作对象, 对输入的Tensor进行计算, 得到一个Tensor对象, 然后通过函数返回值返回该对象
                    所以不管输入还是输出, 都是Tensor, 里面计算的细节被屏蔽了
                而打印Tensor对象, 一共有三个参数, :
                    Tensor{"", shape=(), dtype=}
                    第一个参数(我们称之为 指令名称 ) 就告诉你了这个Tensor变量是通过哪个操作对象得来的 -> 这个我们称之为指令名称
            注意: 打印出来的是张量值, 可以理解成OP当中包含了这个值. 并且每一个op指令, 都可以对应一个唯一的值, 就是指令名称
                命名形式: <OP_NAME>:<int>
                    <OP_NAME>: 是生成该张量的指令的名称
                    <int>: 是一个整形, 它表示该张量在指令的输出个数的索引


            讲义: 一个操作对象(Operation)是TensorFlow图中的一个节点, 可以接受0个或者多个输入Tensor, 并且可以输出0或者多个Tensor
            Operation对象是通过op构造函数创建的

        2 指令名称
            一张图 -> 一个命名空间
            tf会自动为图中每一个指令选择一个唯一名称, 用户也可以指定描述性名称, 使程序阅读起来更轻松
                更改名字:
                每个创建新的tf.Operation(输入, name="") # 指定name参数
                如果指定的是一致的name, 那么tf将会在名称上附加上_ 1, 2, 以便使名称具有唯一性

            也就是同一个图不能有相同的指令名称 但是不同图之间可以有相同的指令名称

2.3 会话
    2.3.1 会话
    一个运行TensorFlow operation的类, 会话中包含了以下两种开启方式.
        tf.Session：用于完整的程序当中
        tf.InteractiveSession：用于交互式上下文中的TensorFlow ，例如shell
            可以用.evel()来查看变量的值, 这种方式只能用在Session中.(例如c_t.eval())
        1. TensorFlow使用tf.Session类来表示客户端程序(通常为python程序, 但是也提供了使用其他语言的类似接口),与c++运行之间的链接
        2. tf.Session对象使用分布式TensorFlow运行时提供对本地计算机中的设备和远程设备的访问权限.

        1）会话是掌握一定资源的, 用完要回收(例如文件) -> 所以使用上下文管理器(with...)
        2）初始化会话对象时的参数如下:
            graph=None # 运行哪一张图, None -> 就是运行默认图
            target：如果将此参数留空（默认设置）, 会话将仅使用本地计算机中的设备。
                可以指定 grpc:// 网址，以便指定 TensorFlow 服务器的地址, 这使得会话可以访问该服务器控制的计算机上的所有设备。
            config：此参数允许您指定一个 tf.ConfigProto以便控制会话的行为。
                例如，ConfigProto协议用于打印设备使用信息, 可以看日志, 每一个操作运行在那一台机器上

        3)run(fetches,feed_dict=None, option=None, run_metadata=None)
            作用: 通过使用sess.run() 来运行我们已经定义好的operation
            参数:
                fetches: 单一的operation, 或者列表, 元组(其他不属于tf的类型不行)
                feed_dict: 参数允许调用者覆盖图中张量的值, 运行时赋值
                    与tf.placeholder搭配使用, 则会检查值的形状是否与占位符兼容

        3 feed操作 -> placeholder提供占位符, run的时候通过feed_dict指定参数
            placeholder是指, 在定义张量的时候我们不知道具体的值是什么, 于是我们使用placeholder占位(仅仅指定类型)
            在运行的时候, 我们再给其赋值.
            例子:
                # 定位占位符.
                a = tf.placeholder(tf.float32, shape=)
                b = tf.placeholder(tf.float32, shape=)
                sum_ab = tf.add(a, b)
                # 开启会话:
                with tf.Session as sess:
                    # 上面定义了定位符, run的时候就要使用feed_dict对元素进行赋值(使用字典形式)
                    # 注意, 赋值的类型要匹配
                    sess.run(sum_ab, feed_dict={a:3.0, b:2.0}
    请注意运行时候的保存类型: error
    RuntimeError: 如果这Session是无效状态(例如已关闭)
    TypeError: 如果fetches或者feed_dict键的类型不合适
    ValueError: 如果fetches或者feed_dict键无效引用Tensor不存在的键

2.4 张量Tensor
    print()
    ndarray与Tensor关系很近
    2.4.1 张量(Tensor)
    tf的张量就是一个n维数组, 类型为tf.Tensor.
        张量 在计算机当中如何存储？
        标量 一个数字                 0阶张量
        向量 一维数组 [2, 3, 4]       1阶张量
        矩阵 二维数组 [[2, 3, 4],     2阶张量
                     [2, 3, 4]]
        ……

        张量 n维数组                  n阶张量

    Tensor具有下面两个非常重要的属性:
        1 张量的类型 -> dtype
        2 张量的形状(阶) -> shape
        创建张量的时候，如果不指定类型
    张量的类型:
        数据类型        Python类型        描述
        DT_FLOAT        tf.float32      32位浮点数
        DT_DOUBLE       tf.float64      64位浮点数
        DT_INT64        tf.int64        64位有符号整数
        DT_INT32        tf.int32        32位有符号整数
        DT_INT8         tf.int8         8位有符号整数
        DT_UINT8        tf.uint8        8位无符号整数
        DT_STRING       tf.string       可变长度的字节数组, 每一个张量元素都是一个字节数组
        DT_BOOL         tf.bool         布尔类型
        DT_COMPLEX64    tf.complex64     由2个32位浮点数组成的复数: 实数和虚数
        DT_QINT32       tf.qint32        用于量化Ops的32位有符号整数
        DT_QINT8        tf.qint8         用于量化Ops的8位有符号整数
        DT_QUINT8       tf.quint8       用于量化Ops的8位无符号整数

        如果不指定类型, 默认: tf.float32
            整型 tf.int32
            浮点型 tf.float32

        张量的阶:
            阶         数学实例         python        例子
            0           纯量          只有大小        s=123
            1           向量          大小和方向       v=[1, 2, 3]
            2           矩阵          数据表         m=[[][]...]
            3           3阶张量        数据立体        t=[ [ [] []... ]... ]
            n           n阶

    2.4.2 创建张量的指令
        固定值张量：
            tf.zeros(shape, dtype=float32, name=None)
                创建所有元素值为0的张量, 此操作返回一个dtype类型,形状为shape和所有元素设置为0的类型的张量.
            tf.zeros_like(tensor, dtype=None, name=None)
                给定tensor定单张量(), 此操作返回tensor与所有元素设置为零相同的类型和形状的张量。
            tf.ones(shape, dtype=tf.float32, name=None)
                创建一个所有元素设置为1的张量。此操作返回一个类型的张量，dtype形状shape和所有元素设置为1。
            tf.ones_like(tensor, dtype=None, name=None)
                给tensor定单张量（），此操作返回tensor与所有元素设置为1 相同的类型和形状的张量。
            tf.fill(dims, value, name=None)
                创建一个填充了标量值的张量。此操作创建一个张量的形状dims并填充它value。
            tf.constant(value, dtype=None, shape=None, name='Const')
                创建一个常数张量。
        随机性张量:
            tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
                从正态分布中输出随机值，由随机正态分布的数字组成的矩阵
            tf.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
                从均匀分布输出随机值。生成的值遵循该范围内的均匀分布 [minval, maxval)。下限minval包含在范围内，而maxval排除上限。

        在我们使用op创建tensor的时候, 这个变量是tensor, 但是我们使用(输出)的时候, 它本质上是ndarray
        ndarray属性的修改
            类型的修改
                1）ndarray.astype(type)
                tf.cast(tensor, dtype)
                    不会改变原始的tensor
                    返回新的改变类型后的tensor
                2）ndarray.tostring() # 转换成为
            形状的修改
                1）ndarray.reshape(shape) # 返回新的数组
                    -1 自动计算形状
                2）ndarray.resize(shape) # 修改原来的数组
        tf中的转换:
            提供了如下一些改变张量中数值类型的函数
                tf.string_to_number(string_tensor, out_type=None, name=None)
                tf.to_double(x, name='ToDouble')
                tf.to_float(x, name='ToFloat')
                tf.to_bfloat16(x, name='ToBFloat16')
                tf.to_int32(x, name='ToInt32')
                tf.to_int64(x, name='ToInt64')
                tf.cast(x, dtype, name=None) # 通用类型转换
                    不会改变原始的tensor
                    返回新的改变类型后的tensor
                其中x是原始的tensor对象, name表示指令名称
        形状改变:
            tf中的张量具有两种形状变换, 动态形状和静态形状
                tf.reshape
                tf.set_shape

                静态形状 -> 初始创建张量时的形状
                1）如何改变静态形状
                    tensor.set_shape(shape)
                    什么情况下才可以改变/更新静态形状？
                        只有在形状没有完全固定下来的情况下, 才可以通过set_shape()进行改变
                            什么情况下才属于形状没有完全固定下来:
                                通过tf.placeholder()得到的Tensor变量, 其shape=(?,?) -> 这就是属于没有完全固定下来的静态形状
                            # 更新"形状未确定"的部分
                            # 但是要注意, 已经定义的阶数仍然不能变, 是几阶就是几阶
                    （1）转换静态形状的时候，1-D到1-D，2-D到2-D，不能跨阶数改变形状
                    （2）对于已经固定或者设置静态形状的张量/变量，不能再次设置静态形状

                2）如何改变动态形状, 什么情况下改变.
                    tf.reshape(tensor, shape) # 动态形状修改, 可以跨越维数进行转换
                    不会改变原始的tensor
                    返回新的改变形状后的tensor, 这里与torch中一样
                    动态创建新张量时，张量的元素个数必须匹配(一致) !, 如果不一致, 则会报错, 跨阶也需要保持元素个一致.

2.4.4 张量的数学运算
    算数运算
    基本数学运算函数
    矩阵运算
    reduce操作 -> 减少张量的各个维度.
    序列索引操作

    可以查看API:https://tensorflow.google.cn/api_docs/python/tf/all_symbols
    看示例


2.5 变量 OP
    TensorFlow 中的一个组件 -> 变量
    TensorFlow变量是表示程序处理的共享持久状态的最佳方法, 变量通过tf.variable OP类进行操作
    变量的特点:
        存储持久化
        可修改值
        可指定被训练

        变量也就是可以: 存储模型参数 -> 有了参数, 就可以构建一个模型. 而变量就是存储这些参数的.
    API:
        tf.Variable(initial_value=None, trainable=True, collections=None, name=None)
            initial_value: 初始化的值,
            trainable: 是否被训练
            collections: 新变量将添加到列出的图的集合中collections.默认为:
                GraphKeys.GLOBAL_VARIABLES, 如果trainable是True变量也被添加到图形集: GraphKeys.TRAINABLE_VARIABLES
            name: 指令名称


    2.5.1 创建变量
        变量需要显式初始化，才能运行值
            init = tf.global_variables_initializer()
            然后在run()方法中传入, init对象, 这样就运行起来了刚才定义的变量
    2.5.2 使用tf.variable_scope()修改变量的命名空间
        (这也是一个上下文管理器, 需要配合with使用)
        使得结构更加清晰, 代码更加模块化.


2.6 高级API -> 就是基础API写出的
    2.6.1 其它基础API -> 提供一些神经网络的组件.
        1. tf.app
            这个模块相当于为TensorFlow进行的脚本提供一个main函数入口, 可以定义脚本运行的flags
        2 tf.image
            TensorFlow 的图像处理操作. 主要是一些颜色变换、变形和图像的编码和解码.
        3 tf.gfile
            这个模块提供了一组文件操作函数。
        4 tf.summary
            用来生成 TensorBoard 可用的统计日志，目前 Summary 主要提供了 4 种类型：audio、image、histogram、scalar
        5 tf.python_io
            用来读写 TFRecords文件
        6 tf.train
            这个模块提供了一些训练器，与 tf.nn 组合起来，实现一些网络的优化计算。
        7 tf.nn
            这个模块提供了一些构建神经网络的底层函数。 TensorFlow 构建网络的核心模块。
            其中包含了添加各种层的函数，比如添加卷积层、池化层等。
    2.6.2 高级API -> 将一些基础的API排列组合然后形成的高级的API
        1 tf.keras
            Keras 本来是一个独立的深度学习库，tensorflow将其学习过来，增加这部分模块在于快速构建模型。
        2 tf.layers
            高级 API，以更高级的概念层来定义一个模型。类似tf.keras。
        3 tf.contrib
            tf.contrib.layers提供够将计算图中的 网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。
        4 tf.estimator
            一个 Estimator 相当于 Model + Training + Evaluate 的合体。
            在模块中，已经实现了几种简单的分类器和回归器，包括：Baseline，Learning 和 DNN。
            这里的 DNN 的网络，只是全连接网络，没有提供卷积之类的。

使用TensorFlow用起来
2.7 案例：实现线性回归
    2.7.1 线性回归原理复习
        1）构建模型
            y = w1x1 + w2x2 + …… + wnxn + b -> 回归函数
        2）构造损失函数 -> 用来优化模型
            均方误差 -> MSE
            OSL(最小二乘法)
        3）优化损失
            正规方程, 梯度下降
    用到的张量运算API:
        矩阵运算: tf.matmul(x, w)
        平方: tf.square(error)
        均值: tf.reduce_mean(error)
        梯度下降优化:
            tf.train.GradientDescentOptimizer(learning_rate=0.01)
                梯度下降优化
                learning_rate: 学习率, 一般为0~1之间比较小的值
                method:
                    minimize(error) -> 最小化
                return: 梯度下降op
    2.7.2 案例：实现线性回归的训练
        准备真实数据
            100样本
            x 特征值 -> 形状 (100, 1)
            y_true 目标值 -> 目标值 (100, 1)
            设满足:
                y_true = 0.8x + 0.7
        假定x 和 y 之间的关系 满足
            y = kx + b
            k ≈ 0.8 b ≈ 0.7

            流程分析：
            (100, 1) * (1, 1) = (100, 1)
            -> y_predict = x * weights(1, 1) + bias(1, 1)
            1）构建模型
                y_predict = tf.matmul(x, weights) + bias
            2）构造损失函数
                error = tf.reduce_mean(tf.square(y_predict - y_true))
            3）优化损失
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

            5 学习率的设置、步数的设置与梯度爆炸
                学习率越越大, 训练到较好结果的步数越小; 学习率越小, 训练到较好的结果的步数越大.
                    我的理解: 学习率相当于步长, 越大, 说明逼近的越快.
                但是学习过大会出现梯度爆炸现象. 关于梯度爆炸/梯度消失?
                    在极端情况下, 权重的值变得非常大, 以至于溢出, 造成NaN
                    那么如何解决梯度爆炸的问题(深度神经网络当中更容易出现)
                    1. 重新设计网络
                    2. 调整学习率
                    3. 使用梯度截断
                    4. 使用激活函数.



    2.7.3 对线性回归 -> 增加其他功能
        1 增加变量显示
            1）创建事件文件
            2）收集变量
            3）合并变量
            4）每次迭代运行一次合并变量
            5）每次迭代将summary对象写入事件文件

        2 增加命名空间
            with tf.variable_scope(""):

            -> 使得结构更加清晰
        3 模型的保存与加载
            saver = tf.train.Saver(var_list=None,max_to_keep=5)
            1）实例化Saver
            2）保存
                saver.save(sess, path)
            3）加载
                saver.restore(sess, path)

        4 命令行参数使用
            1）tf.app.flags. 定义
                它支持应用从命令行接受参数, 可以用来指定集群配置等. 在tf.app.flags下面有各种定义参数的类型
                DEFINE_string(flag_name, default_value, docstring) -> 传入的是字符串
                DEFINE_integer(flag_name, default_value, docstring) -> 传入的是整数
                DEFINE_boolean(flag_name, default_value, docstring)
                DEFINE_float(flag_name, default_value, docstring)
            参数:
                flag_name: 名字
                default_value: 默认值, 没有使用该参数的时候, 默认是默认值
                docstring: 描述

            tf.app.flags.DEFINE_integer("max_step", 0, "训练模型的步数")
            tf.app.flags.DEFINE_string("model_dir", " ", "模型保存的路径+模型名字")

            2）FLAGS = tf.app.flags.FLAGS
                tf.app.flags在flags有一个FLAGS标志, 他在程序中可以调用到我们前面具体定义的flag_name
                通过FLAGS.max_step调用命令行中传过来的参数.

            3、通过tf.app.run()启动main(argv)函数

"""
