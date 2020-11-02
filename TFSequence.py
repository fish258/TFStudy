import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# 1. ####################################################################
# Define Sequential model with 3 layers
# model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3")# 没有激活函数
#     ]
# )
# # Call model on a test input
# x1 = tf.ones((3, 3))
# y1 = model(x1)
# model.summary()

# 2. ####################################################################
# 等同于下边这个
# create 3 layers
# layer1 = layers.Dense(2, activation="relu", name="layer1")
# layer2 = layers.Dense(3, activation="relu", name="layer2")
# layer3 = layers.Dense(4, name="layer3")
# # Call layers on a test input
# x2 = tf.ones((3, 3))
# y2 = layer3(layer2(layer1(x2)))

# 3. ####################################################################
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4,name="layer3"))
shape = (3,3)
model.build(shape)
model.summary()
print(len(model.layers))
print(model.weights)


# 删除一层
# model.pop()
# model.build(shape)
# model.summary()
# print(len(model.layers))
