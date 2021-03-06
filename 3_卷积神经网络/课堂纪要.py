"""
卷积神经网络原理
    卷积神经网络介绍
    卷积神经网络 - 结构 大体上由四层结构组成
        卷积层 (最重要的)
        激活函数 (激活层)
        池化层
        全连接层 (分类用)
快速上手: Mnist数据集 - 使用卷积神经网络
案例: 使用卷积神经网络: 验证码识别案例

4.1 卷积神经网络简介:
    1）卷积神经网络与传统多层神经网络对比:
        传统意义上的多层神经网络是只有输入层、隐藏层、输出层。
            其中隐藏层的层数根据需要而定，没有明确的理论推导来说明到底多少层合适
        卷积神经网络CNN，在原来多层神经网络的基础上，加入了更加有效的特征学习部分，
            具体操作就是在原来的全连接的层前面加入了部分连接的卷积层与池化层。
            卷积神经网络出现，使得神经网络层数得以加深，深度学习才能实现
        卷积神经网络:　结构:
            输入层
            隐藏层
                卷积层
                激活层
                池化层
                    pooling layer
                    subsample
                全连接层
            输出层

    2）发展历史:
        最终结果的好坏, 与网络结构有很大关系.
        发展:
            网络结构加深
            加强卷积功能
            从分类到检测
            新增功能模块

    3）卷积网络在ImageNet比赛 错误率:
        反正就是很厉害就对了
        最厉害的还是残差神经网络

4.2 卷积神经网络原理:
    神经网络(neural networks)的基本组成包括输入层, 隐藏层, 输出层. 而卷积神经网络的特点在于隐藏层
    分为卷积层和池化层(poopling layer, 又叫下采样层) 以及激活层.
    卷积神经网络 - 作用
        卷积层:
            拿着一个过滤器(小窗口)通过在原始图片上平移, 来提取特征.
            即: 通过在原始图像上平移来提取特征
        激活层:
            加一些激活函数, 可能是线性的, 可能是非线性的
            即: 增加非线性分割能力
        池化层(下采样层):
            降低模型复杂度, 减少模型参数. -> 可以防止过拟合
            即: 减少学习的参数，降低网络的复杂度（最大池化和平均池化）
        全连接层:
            最后的输出层, 进行损失计算并且输出分类结果.
            即: 进行分类

# 下面开始分别说一下各个层的作用
4.2.2 卷积层（Convolutional Layer）
    卷积神经网络中的每层卷积层由若干卷积单元(卷积核)组成, 每个卷积单元的参数都是反向传播算法最佳化得到的.
    卷积运算的目的是"特征抽取", 第一层卷积层可能只能提取一些低级的特征和边缘, 线条和角等层级, 更多层的网络能从低级特征中迭代提取更复杂的特征.

    卷积层中主要的一个结构就是卷积核
    卷积核 -> filter - 过滤器 - 卷积单元 - 模型参数 (很多名字)
        Filter的四大要素: 卷积核的个数, 卷积核的大小, 卷积核的步长, 卷积核零填充大小
        个数:
            那么如果在某一层结构中, 不只是一个人观察, 多个人(卷积核)一起去观察, 那么就得到了多张观察结果
            - 不同的卷积核带的权重和偏置都不一样, 即随机初始化的参数


        大小:卷积核大小:
            例如: 1*1 3*3 5*5, 通常卷积核大小选择这些大小, 是经过研究人员实验证明比较好的效果.
            每一个格子有权重值(可能符合高斯等分布), 乘以图片的像素值, 加上偏置
            然后我们在图像上平滑移动, 加权计算, 就得到一幅新的图像, 也就是对原图像进行特征抽取了
            卷积如何计算？
                输入:
                    原始图像:5*5*1
                    filter形状: 3*3*1
                    步长 1 (每次移动的距离)是1
                输出 (原始图像经过filter移动后):
                    3*3*1
        步长: 卷积核平移的步长, 影响最后输出的结果
            输入
                原始图像:5*5*1
                filter 3*3*1
                步长2
            输出
                2*2*1 最后的结果

        零填充的大小:
            我们已经得出输出结果的大小由大小和步长决定, 但是只有这些么?
            还有一个就是零填充.Filter观察窗口的大小和移动步长有时会导致超过图片像素宽度.
            此时我们该怎么办? 一种方法: 补0(零填充), 另一种方法, 直接丢弃

            显然, 零填充的做法时更为普遍的:
            零填充就是在图片像素外围填充一圈值为0的像素


    6 总结-输出大小计算公式 ***
        如果已知输入图片形状, 卷积和数量, 卷积核大小, 以及移动步长, 那么输出图片形状土豪确定?
        输入体积大小: H1 * W1 * D1
        四个超参数:
            Filter数量: K
            Filter大小: F (F*F, 就是卷积核的大小)
            步长: S
            零填充大小: P
        输出体积大小: H2 * W2 * D2
            H2 = (H1 - F + 2P)/S + 1
            W2 = (W1 - F + 2P)/S + 1
            D2 = K
    7 多通道图片如何观察:
    如果是一张彩色图片, 那么就有三种表分别为R, G, B. 原本每个人需要带一个3*3或者其他大小的卷积核,
    现在需要带三张3*3的权重和一个偏置, 总共就是27个权重. 最终每个人还是得出一张结果
        输入图片: 5*5*3
        零填充=1, 所以图片: 7*7*3
        Filter: 3*3*3  2个filter
        步长: 2
        输出: 3*3*2
        仔细看一下这个过程
        我们一组卷积核是3个3*3, 然后对应三个通道上的图片, 得到3个经过卷积核的结果, 然后再把3个结果线性加权后得到最终的一个3*3*1的结果
        有两个filter,所以最终得到的是两个3*3*1的结果, 也就是3*3*2

            原始图片5*5*3 filter 3*3*3 + bias 2个filter 步长2
            H1=W1=7
            D1=3
            K=2
            F=3
            S=2
            P=1 # 填充了一圈

            H2=(5-3+2)/2+1=3 = W2
            D2=k=2
        输出
            3*3*2 结果一致.

    在TensorFlow中使用卷积神经网络:
    卷积网络API
        tf.nn.conv2d(input, filter, strides=, padding=)
        功能: 计算给定4-D input和filter张量的2维卷积.
        参数: 其实对应的就是前面filter的四要素
            input：给定的输入张量, 具有[batch,heigth,width,channel], 类型为float32,54
                也就是输入图像
                要求: 形状[batch,heigth,width,channel]
                      类型为float32,64
            filter: 卷积核, 也是训练出来的模型参数
                    指定过滤器的权重数量, [filter_height, filter_width,in_channel, out_channels]
                权重weights & bias, 模型参数用变量保存
                # 形状: [filter_height, filter_width,in_channel, out_channels]
                # [filter高, filter宽, 输入图片的通道数, 输出图片的通道数(也是要素D2=K)]
                变量initial_value=random_normal(shape=[F, F, 3/1, K]) # 4阶张量
            strides: 步长
                strides = [1, stride, stride, 1] 步长
                例子: 步长 1, 取1 的时候比较多,
                    [1, 1, 1, 1] 横着移动1像素, 竖着移动1像素
            padding: 零填充 有两种方式"SAME", "VALID"
                "SAME"：越过边缘取样, 相当于加上一些0填充
                    取样的面积和输入图像的像素宽度一致.
                    不过公式简化了: ceil(H/S)
                    无论过滤器的大小是多少, 零填充的数量由API自动计算
                "VALID"：不越过边缘取样, 如果越过边缘了, 那我们整块像素就都不要了
                    取样的面积小于输入图像的像素宽度, 则不填充, 直接放弃
    总结:
    1) 掌握filter要素的相关计算公式
    2) filter大小
        1x1，3x3，5x5
       步长 取1比较普遍
       过滤器的个数不一定, 不同结构选择不同
    3) 每个过滤器会带有若干权重和1个偏置

    4.2.3 激活函数 -> 激活层
    随着神经网络的发展, 大家发现原有的sigmoid等激活函数并不能达到好的效果, 所以采取新的激活函数.
    为什么采用新的激活函数:
        sigmoid: 1/(1+e^-x)
        缺点:
            1）计算量相对大
            2）反向传播的过程中容易出现梯度消失(梯度=0)的情况
            3）输入的值的范围[-6, 6], 当绝对值>6的时候,变化不明显, 失效了
        Relu的好处: max(0, x), 小于0的时候为0, 大于0的时候为x
            1）计算速度快
            2）有效解决了梯度消失收集变量
            3）图像没有负的像素值, 所以正好符合ReLU
        还有很多其他的激活函数, 用到时候可以查查.
        使用playground查看不同激活函数的效果
       API:
             tf.nn.relu(features, name=None)
             features: 卷积后加上偏置的结果
             return: 经过激活函数之后的结果
        
    4.2.4 池化层(Poling) -> 减少学习的参数, 降低网络的复杂度
        Polling层主要的作用是特征提取, 通过去掉Feature Map中不重要的样本, 进一步减少参数数量.Polling的方法有很多, 通常采用最大池化
        max_poling: 取池化窗口的最大值
        avg_poling: 取池化窗口的平均值

        池化, 实际上利用了图像上像素点之间的联系
        池化层计算:
            池化层也有窗口的大小以及移动步长, 那么之后的输出大小怎么计算?
            计算公式与计算卷积时的一致.
            计算: 224 * 224 * 64, 池化窗口为2, 步长为2的输出结果, 零填充是0:
            H2 = (224 - 2)/2 + 1 = 112 - 1 + 1 = 112
            D2 = D1 = 64.

        API:
        tf.nn.max_pool(value, ksize=, strides=, padding=, name=None)
            功能: 输入上执行最大池数
            value:
                4-D Tensor形状[batch, height, width, channels]
                其中channels并不表示通道数了, 而是经过卷积层时, 有多少个filter就有多少个channels
            ksize：
               池化窗口大小，[1, ksize, ksize, 1]
            strides:
                步长大小，[1, strides, strides, 1]
            padding：零填充方式算法. 默认是SAME
                "SAME": 越过边缘取样
                "VALID": 不越过边缘取样
    4.2.5 全连接层(Full Connection)
        前面的卷积和池化相当于做特征工程, 最后的全连接层在整个卷积神经网络中起到"分类器"的作用.


4.3 案例：CNN-Mnist手写数字识别
    4.3.1 网络设计
            我们有两个卷积大层,
            每个卷积大层, 又分为: 卷积层, 激活层, 池化层
            最后有一个全连接层, 作为分类问题的输出
        我们不仅需要清楚网络结构设计, 还要清楚形状怎么变化的:
        原始图像: 784长度
        第一个卷积大层：
            卷积层：
                32个filter 大小5*5, 步长：1, padding="SAME"
                 tf.nn.conv2d(input, filter, strides=, padding=)
                 input：需要输入图像形状 [None, 28, 28, 1]
                     要求：形状[batch,heigth,width,channel]
                     类型为float32,64
                 filter: 卷积核其实就是模型参数, 底层其实就是weight和bias
                        所以我们用变量来保存这些变量.
                     weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32])) # 32个filter
                     bias = tf.Variable(initial_value=tf.random_normal(shape=[32])) # 每个filter带一个bias, 所以这里初始化成32
                     变量initial_value=random_normal(shape=[F, F, 3 or 1, K]) # 这就是形状
                 strides:
                     步长 1
                      [1, 1, 1, 1]
                 padding: “SAME”, 算法,  自动选取零穿越像素数目, SAME算法, 形状不变
                     “SAME”：越过边缘取样, 形状保持一致
                     “VALID”：不越过边缘取样, 图片形状变小
                 形状变化:
                 输出形状:
                    [None, 28, 28, 1]
                 输出形状：
                    [None, 28, 28, 32]
                    因为是SAME, 所以高度, 宽度不变, 又有32个filter, 所以channel变为32
            激活：不改变形状
                Relu
                tf.nn.relu(features)
            池化：
                输入形状：[None, 28, 28, 32]
                大小2*2 步长2 (习惯) 零穿越: 0
                H2 = (28 - 1)/2 + 1 向上取整 -> 13.5 -> 14
                输出形状：[None, 14, 14, 32]

        第二个卷积大层：输入: 输出形状：[None, 14, 14, 32]
            卷积层：
                64个filter 大小5*5, 步长：1, padding="SAME"
                输入：[None, 14, 14, 32]
                tf.nn.conv2d(input, filter, strides=, padding=)
                input：[None, 14, 14, 32]
                    要求：形状[batch,heigth,width,channel]
                    类型为float32,64
                filter:
                    weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64]))
                                # 32 是输入图像的通道数, 64是这层的filter数
                    bias = tf.Variable(initial_value=tf.random_normal(shape=[64])) # 多少个filter多少个bias
                    变量initial_value=random_normal(shape=[F, F, 3/1, K])
                strides:
                    步长 1
                     [1, 1, 1, 1]
                padding: “SAME”
                    “SAME”：越过边缘取样
                    “VALID”：不越过边缘取样
                输出形状：
                [None, 14, 14, 64]
            激活：形状不变
                Relu
                tf.nn.relu(features)
            池化：
                输入形状：[None, 14, 14, 64]
                大小2*2 步长2
                同理:
                输出形状：[None, 7, 7, 64]

        全连接 -> 分类
            线性加权, 矩阵相乘 -> 所以需要改变形状(改变成2阶, 这样是一个矩阵)
            tf.reshape()
            [None, 7, 7, 64] -> [None, 7*7*64] (跨越阶数, 所以使用动态形状转变)
            [None, 7*7*64] * weight = [None, 10] -> 最终的结果. 也就是[样本数, 结果]
                所以可以推断出: weight的形状: [7*7*64, 10]
            所以操作:
            y_predict = tf.matmul(pool2, weithts) + bias

        调参->提高准确率？
        1）调整学习率
        2）随机初始化的权重、偏置的值, random_normal(mean, standard)
        3）选择好用的优化器: 优化损失
        4）调整网络结构(包括网络结构, 激活函数)

4.4 网络结构与优化
    4.4.1 网络的优化和改进
        初始参数大小调整
            模型参数的放大, 缩小
        使用改进版本的SGD算法
        对于深度网络使用batch normalization 或者 droupout .-> 防止过拟合
            通过加入这batch normalization可以使这一层输出的权重系数分布保持一致(同样的规律内)
            droupout是直接让某些神经元失效, 使某些模型参数失效, 降低模型复杂夫, 避免过拟合情况
        或者使用更高级的API

    4.4.2 卷积神经网络的拓展了解
        1 常见网络模型:
            google le net -> inception V3, inception V4
            from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
        2 卷积网络其它用途
            yolo: GoogleNet + bouding boxes
            SSD:




4.5 实战：验证码图片识别
    验证码识别实战.
        1）数据集
            图片数据, 和一个csv存储的label
            如何识别: 分割? 还是整体识别?
                这个数据集当中的数字还是比较规整的, 但是并不是所有的验证码图片都是这么规整的
                所以除非能够自适应切割图片, 否则切割不具有通用性
            回顾:
                一张手写数字的图片 -> 0~9之间的某一个数 即:一个样本对应一个目标值 -> softmax交叉熵
                目标值:是[0,0,1,0……] one-hot编码形式

                那么: 图片1 -> NZPP 即:一个样本对应4个目标值 -> sigmoid交叉熵
                    NZPP -> [13, 25, 15, 15] 先对26个字母编码, 然后转换成one-hot编码:
                    -> [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                    所以形状:其实是[4, 26]:
                        [
                            [26...]
                            4..
                        ]
                        这样就方便我们后期对结果值的一个预测了。
                编码成onr-hot编码API:
                    tf.one-hot(indices, depth, axis=None, name=None)
                        indices: 需要编码的张量
                        depth: one_hot编码的深度, 样本有几种可能性, 这里例子是26
                        axis: 填充的维度, 默认是-1
                那么如何衡量损失呢?

        2）对数据集中
            特征值: 6000张图片
            目标值: csv文件中的label
            怎么用
        3）如何分类？
            如何比较输出结果和真实值的正确性？
            如何衡量损失？
                手写数字识别案例：
                    线性加权 -> softmax + 交叉熵 衡量损失.
                    如果是该案例: [4, 26] -> [4*26]
                    如果计算概率值时仍然使用softmax函数, 并对这两个104个元素的一阶张量计算交叉熵损失,
                        会存在损失值过大并且无法减小的问题.(因为softmax+交叉熵,只能优化一个地方的值,而无法做到全局优化)
                        由于softmax针对所有104个输出的logits计算输出概率,和为1, 最好的结果是真实值编码为1的4个位置对应的预测概率值为0.25.
                        所以,在优化迭代过程中, 如果要使概率进一步提供, 势必造成其他三个真实值编码为1的位置对应的概率值下降.
                        所以softmax + 交叉熵, 只适用于一个样本对应一个目标值
                sigmoid交叉熵: 适用于: 每个样本类别独立并且不互相排斥的离散分类任务中的损失值.
                    API:
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=None, logits=None)
                            labels: 真实值, 为ont-hot编码形式和logts一样的形状
                            logits: 预测值, logits值输出层的加权计算结果

            准确率如何计算？
                核心：对比真实值和预测值最大值所在位置, 然后求一个平均值/.
                回顾: 手写数字识别案例
                    y_predict[None, 10]
                    tf.argmax(y_predict, axis=1) #
                本案例中:
                    y_predict[None, 4, 26]
                    tf.argmax(y_predict, axis=2 or  -1) # 需要以26那一维度进行激素,
                返回的结果, 因为这是一个样本对应多个值, 只有全对了, 才能说对了. 所以用all(逻辑多与)
                    [True,
                    True,
                    True,
                    False] -> tf.reduce_all() -> False

        4）流程分析
            1）读取图片数据
                读取:
                key, value = reader.read()
                这个key就是文件名: 也就是编码, 然后去csv中找对应编码, 返回的就是验证码的真实值
                    即: key(filename) -> 标签值

            2）解析csv文件，将标签值处理成数字的形式, 方面后面ont_hot编码
                NZPP -> [13, 25, 15, 15]
                最终结果:
                file_num            chars           labels
                0                   NZPP            [13, 25, 15, 15]
            3）将filename和标签值联系起来
                每个样本对应什么值
            4）构建卷积神经网络 return:-> y_predict
                使用我们前面构建好的模型
            5）构造损失函数

            6）优化损失
            7）计算准确率
            8）开启会话、开启线程
        5）代码实现

    流程分析:
        特征值: 6000张图片
        目标值: csv文件中的label
        怎么用, 需要一一对应起来
        -> 构建模型
        1. 读取图片数据:
            key, value = read(file_queue)
            key: 文件名 -> 对应的编号, 可以去csv文件中对比 -> 目标值
            value: 对应的图片值
        2. 构建一个文件名和标签值对应的表格:
        解析csv文件, 建立一个文件名和标签值对应的表格
        类似于:
            file_num            chars           labels
            0                   NZPP            [13, 25, 15, 15]
        3. 然后根据训练时得到的图片的文件名, 得到对应的标签值
            将一个样本(图片)的特征值和目标值一一对应
            通过文件名查表
        4. 构建卷积神经网络模型: -> y_predict
        5. 计算sigmold交叉熵损失
        6. 优化损失
        7. 计算准确率
        8. 开启会话, 开启线程
            不断迭代, 训练.


"""