'''
# Time   : 2020/10/21 16:55
# Author : junchaoli
# File   : model.py
'''

import tensorflow as tf
import tensorflow.keras.backend as K

class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):# 只初始化一些超参数.和模型定义的超参数.
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
    '''
    简单翻译，就是说官方推荐凡是tf.keras.layers.Layer的派生类都要实现__init__()，build(), call()这三个方法
    __init__()：保存成员变量的设置
    build()：在call()函数第一次执行时会被调用一次，这时候可以知道输入数据的shape。返回去看一看，果然是__init__()函数中只初始化了输出数据的shape，而输入数据的shape需要在build()函数中动态获取，这也解释了为什么在有__init__()函数时还需要使用build()函数
    call()： call()函数就很简单了，即当其被调用时会被执行。
    ————————————————
    版权声明：本文为CSDN博主「_吟游诗人」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/qq_32623363/article/details/104128497
    '''
    def build(self, input_shape):  # 所以build 的参数只有一个就是input_shape.用来初始化一些需要根据input_shape来用的变量.====推荐初始化参数都在这里面进行.
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k) # 算交叉项使用的. https://zhuanlan.zhihu.com/p/342803984 2.3里面写的公式. 推导都不用看,就是个简单的上三角矩阵公式而已.
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)

class FM(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.fm = FM_layer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output
