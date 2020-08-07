import tensorflow as tf

# Keras 模型以类的形式呈现，我们可以通过继承 tf.keras.Model 这个 Python 类来定义自己的模型
# 在继承类中，我们需要重写 __init__() （构造函数，初始化）和 call(input) （模型调用）两个方法，同时也可以根据需要增加自定义的方法
class MyModel(tf.keras.Model):
    def __init__(self):
        super.__init__()

        def call(self, inputs):
            return ""
