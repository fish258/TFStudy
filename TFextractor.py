import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 定义一个模型
initial_model = keras.Sequential(
    [
        layers.Input(shape=(3,3)),
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3")                       # 没有激活函数
    ]
)

# 可以提取每层的输出（下一层的输入）
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers] # list of 每层的输出
)

# Call feature extractor on test input.
x = tf.ones((3,3))
features = feature_extractor(x)
for i in range(len(features)):
    print("Tensor ", i, " :",features[i])












