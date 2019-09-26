import pandas as pd
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
pd.set_option('display.max_columns', 1024)  # 设置最大显示列数

# 在计算图个g1中定义变量c，并将变量c初始化为0
a = tf.constant([1., 2.], name="a")
b = tf.constant([2.0, 3.0], name="b")
c = tf.add(a, b, name="c")
v1 = tf.Variable([1.0, 2.3], name="v1")
v2 = tf.Variable(55.5, name="v2")
feature_embeddings = tf.Variable(
    tf.random_normal([3, 2], 0.0, 0.01),
    name='feature_embeddings')
feature_bias = tf.Variable(tf.random_normal([3, 1], 0.0, 1.0), name='feature_bias')

# 在计算图g1中读取变量c
with tf.Session() as sess:
    # 初始化变量
    # tf.initialize_all_variables().run()
    # tf.train.Saver().save(sess, save_path="F:/python/model/ckpt/")

    tf.train.Saver().restore(sess, save_path="F:/python/model/ckpt/")
    i, predict_data,feat_i,f1,f2,yd,e,r,f11 = sess.run([out, outj,feat_index, fm_first_order, fm_second_order, y_deep,e0,reshaped_feat_value,f1],
                               feed_dict={feat_value: [(1, 1, 1, 1.000000, 1, 1), (1, 1, 1, 1.000000, 1, 1),
                                                       (1, 1, 1, 1.442695, 1, 1)],
                                          feat_index: [(981, 867, 2157, 2156, 948, 0), (982, 868, 2158, 2156, 949, 1),
                                                       (982, 869, 2159, 2156, 949, 1)]}
                               )
    g1 = tf.get_default_graph()
    print(predict_data)
    print("默认格式模型predict_data", e)
    print("默认格式模型predict_data", f11)
    print("默认格式模型predict_data", r)

  



