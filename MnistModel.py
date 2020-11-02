import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as tfds

# 加载数据集
mnist = tfds.mnist
data = mnist.load_data() # 是一个两个tuple的tuple，在子tuple中还是两个tuple
(x_train, y_train), (x_test, y_test) = data  # 60000*28, 60000*1, 10000*28, 10000*1
x_train,x_test = x_train/255.0, x_test/255.0

# 构建model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print(x_train.shape)
model.fit(x_train, y_train, epochs=5) # 以后这可以加batch_size

# 测试模型
model.evaluate(x_test,  y_test, verbose=2)

feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers],
)
a = feature_extractor(x_train[0:2])
print(a[-1][0])
print(a[-1][0][tf.argmax(a[-1][0])])
print(tf.argmax(a[-1][0]))

plt.imshow(x_train[0])
plt.show()