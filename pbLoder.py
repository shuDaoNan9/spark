import pandas as pd
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
pd.set_option('display.max_columns', 1024)  # 设置最大显示列数

with tf.Session() as sess:
        with gfile.FastGFile("F:/python/model/" + 'model.pb', 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图
        # print(sess.graph.as_graph_def().node)
        # 需要有一个初始化的过程
    # sess.run(tf.global_variables_initializer())
    # 需要先复原变量 feature_embeddings
    print(sess.run('feature_embeddings:0'))
    print(sess.run('feature_bias:0'))
    print(sess.run('concat_bias:0'))
    print(sess.run('concat_projection:0'))

    eb = sess.graph.get_tensor_by_name('e0:0')  # 此处的x一定要和之前保存时输入的名称一致！
    f11 = sess.graph.get_tensor_by_name('f1:0')  # 此处的x一定要和之前保存时输入的名称一致！
    r = sess.graph.get_tensor_by_name('r0:0')  # 此处的x一定要和之前保存时输入的名称一致！
    feat_value = sess.graph.get_tensor_by_name('feat_value:0')  # 此处的x一定要和之前保存时输入的名称一致！
    feat_index = sess.graph.get_tensor_by_name('feat_index:0')  # 此处的y一定要和之前保存时输入的名称一致！
    out = sess.graph.get_tensor_by_name('out:0')  # 此处的op_to_store一定要和之前保存时输出的名称一致！
    outj = sess.graph.get_tensor_by_name('add_out:0')  # 此处的op_to_store一定要和之前保存时输出的名称一致！
    predict_data, i,feat_i,e,f,r0 = sess.run([out, outj,feat_index, eb,f11, r],
                               feed_dict={feat_value: [(1, 1, 1, 1.000000, 1, 1), (1, 1, 1, 1.000000, 1, 1), (1, 1, 1, 1.442695, 1, 1)],
                                          feat_index: [(981, 867, 2157, 2156, 948, 0), (982, 868, 2158, 2156, 949, 1),  (982, 869, 2159, 2156, 949, 1)]}
                               )
    print(i)
    print("二进制模型predict_data", e)
    print("二进制模型predict_data", f)
    print("二进制模型predict_data", r0)
