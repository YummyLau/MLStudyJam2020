import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

# 线性模型

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),  # 权重矩阵 kernel，对应于 f(AW + b) 中的 W，初始化为 0
            bias_initializer=tf.zeros_initializer()  # 偏置向量 bias，对应于 f(AW + b) 中的 b，初始化为 0
        )

    def call(self, input):
        output = self.dense(input)
        return output


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    # 自动求导机制，使用求导记录器，记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = model(X)  # 损失函数
        loss = tf.reduce_mean(tf.square(y_pred-y))  # 对所有输入张量的所有元素进行求和
    grads = tape.gradient(loss,model.variables)  # 损失失函数关于自变量（模型参数）的梯度
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
