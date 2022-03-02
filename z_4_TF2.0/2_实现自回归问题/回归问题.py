import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
import sklearn




def regression():
    # 生成网络模型
    model = keras.Sequential() # 生成网络
    model.add(layers.Dense(16)) # 第一层
    model.add(layers.Dense(32))
    model.add(layers.Dense(1))

    # 配置模型
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.01), # 选择反向传播的优化方式
        loss="mean_squared_error" # 选择计算损失的方式
    )

    # 训练
    # model.fit(data, labels, validation_data=)


if __name__ == '__main__':
    regression()
