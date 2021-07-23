'''
# Time   : 2020/10/21 17:51
# Author : junchaoli
# File   : train_lstm.py
'''

from model import FM
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

import argparse
parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('-k', type=int, help='v_dim', default=8)
parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
args=parser.parse_args()

if __name__ == '__main__':
    file_path = 'train.txt'
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)

    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg

    model = FM(k, w_reg, v_reg)
    optimizer = optimizers.SGD(0.01)
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    # model.fit(train_dataset, epochs=200)
    # print(model.evaluate(X_test, y_test))
    # model.summary()

    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:  # 这个是tf2.0加入的eager模式,跟pytorch一样可以逐行debug了.很香. 2.0之后都推荐用这个写法来写.也是tf2.0默认的写法
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables) # 把loss 进行求梯度. 固定写法.
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))# 也是固定写法.

    #评估
    pre = model(X_test) # 跟pyorch不同,不用写model.eval().
    pre = [1 if x>0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))#其实没啥太大客观性.这个数据0,1标签分布差别太大了.绝大部分都是0.有机会拿个大一点的测测.





