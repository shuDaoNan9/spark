#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
pd.set_option('display.max_columns', 100)  # 设置最大显示列数

TRAIN_FILE = "D:\\data\\input\\ml-1m\\"
model_path="D:\\data\\output\\model\\dfm"
rec_path="D:\\data\\output\\rec"
fdict_path = "D:\\data\\output\\feature_dict"

NUMERIC_COLS = ['age']
IGNORE_COLS = ['ratings','timeStamp']
topN=1000
import pandas as pd
import numpy as np
users_Name=['user_id','gender','age','work','zip']
ratings_Name=['user_id','movie_id','ratings','timeStamp']
movie_Name=['movie_id','title','calss']
ratings=pd.read_table(TRAIN_FILE+'ratings.dat',sep='::',header=None,names=ratings_Name)
movies=pd.read_table(TRAIN_FILE+'movies.dat',sep='::',header=None,names=movie_Name)
users=pd.read_table(TRAIN_FILE+'users.dat',sep='::',header=None,names=users_Name)
print('评分表记录数：',len(ratings))
print('**********评分表前五条记录**********')
print(ratings.head(5))

"""模型参数"""
dfm_params = {
    "decay": 0.83,
    "learn_rate_step":976,  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE
    "threads":6,
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 10,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 1,
    "batch_size": 1024,
    "learning_rate": 0.0005,
    # "learning_rate":0.001,  0.1-1 0.01-10 0.002-10 0.001-10(然loss从2变3了，似乎是个别异常数据拉高的)
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": 'gini_norm',
    "random_seed": 3
}

dfTrain = ratings.join(users.set_index('user_id'),on='user_id').join(movies.set_index('movie_id'),on='movie_id')
# dfTrain = pd.read_csv(TRAIN_FILE,nrows=1000000)#.tail(10000)
print(dfTrain.head(5))
print(dfTrain.tail(5))

data_rows=len(dfTrain)
item_df=movies
user_df=users

feature_dict = {}
total_feature = 0
for col in dfTrain.columns:
    if col in IGNORE_COLS:
        continue
    elif col in NUMERIC_COLS:
        feature_dict[col] = total_feature
        total_feature += 1
    else:
        unique_val = dfTrain[col].unique()
        feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
        total_feature += len(unique_val)
print(total_feature)
# with open(fdict_path, 'w+',encoding='utf-8') as f:
#                 f.write(str(feature_dict) )


"""
对训练集进行转化
"""
# with tf.device('/cpu:0'):
train_y = dfTrain[['ratings']].head(500000).values.tolist()
train_feature_index = dfTrain.head(500000).copy()
train_feature_value = dfTrain.head(500000).copy()
# train_y = dfTrain[['ratings']].tail(10000).values.tolist()
# train_feature_index = dfTrain.tail(10000).copy()
# train_feature_value = dfTrain.tail(10000).copy()

for col in train_feature_index.columns:
    if col in IGNORE_COLS:
        train_feature_index.drop(col,axis=1,inplace=True)
        train_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        train_feature_index[col] = feature_dict[col]
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1
print(train_feature_index.head(5))
print(train_feature_value.head(5))
print(train_feature_index.tail(5))
print(train_feature_value.tail(5))

dfm_params['feature_size'] = total_feature  # 254
dfm_params['field_size'] = len(train_feature_index.columns)  # 37
print(len(train_feature_index.columns))

def get_batch(train_feature_index,train_feature_value,train_y, batch_size):
    with tf.device('/cpu:0'):
        print(len(train_feature_index),len(train_y))
        input_queue = tf.train.slice_input_producer([train_feature_index,train_feature_value,train_y],num_epochs=dfm_params['epoch'], shuffle=True )
        i_batch, v_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=dfm_params['threads'], capacity=32, allow_smaller_final_batch=False)
        print(v_batch)
        return i_batch, v_batch,label_batch


"""开始建立模型"""
feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
outj = tf.zeros(shape=[1,1],dtype=tf.float32, name='outj')

"""建立weights"""
weights = dict()

# embeddings
weights['feature_embeddings'] = tf.Variable(
    tf.random_normal([dfm_params['feature_size'], dfm_params['embedding_size']], 0.0, 0.01),
    name='feature_embeddings')
weights['feature_bias'] = tf.Variable(tf.random_normal([dfm_params['feature_size'], 1], 0.0, 1.0), name='feature_bias')

# deep layers
num_layer = len(dfm_params['deep_layers'])
input_size = dfm_params['field_size'] * dfm_params['embedding_size']
glorot = np.sqrt(2.0 / (input_size + dfm_params['deep_layers'][0]))

weights['layer_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(input_size, dfm_params['deep_layers'][0])), dtype=np.float32
)
weights['bias_0'] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][0])), dtype=np.float32
)

for i in range(1, num_layer):
    glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][i - 1] + dfm_params['deep_layers'][i]))
    weights["layer_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(dfm_params['deep_layers'][i - 1], dfm_params['deep_layers'][i])),
        dtype=np.float32)  # layers[i-1] * layers[i]
    weights["bias_%d" % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][i])),
        dtype=np.float32)  # 1 * layer[i]

# final concat projection layer

if dfm_params['use_fm'] and dfm_params['use_deep']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
elif dfm_params['use_fm']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size']
elif dfm_params['use_deep']:
    input_size = dfm_params['deep_layers'][-1]

glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                           dtype=np.float32, name='concat_projection')
weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32, name='concat_bias')

"""embedding"""
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)

reshaped_feat_value = tf.reshape(feat_value, shape=[-1, dfm_params['field_size'], 1])

embeddings = tf.multiply(embeddings, reshaped_feat_value)

"""fm part"""
fm_first_order = tf.nn.embedding_lookup(weights['feature_bias'], feat_index)
fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order, reshaped_feat_value), 2)

summed_features_emb = tf.reduce_sum(embeddings, 1)
summed_features_emb_square = tf.square(summed_features_emb)

squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

fm_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

"""deep part"""
y_deep = tf.reshape(embeddings, shape=[-1, dfm_params['field_size'] * dfm_params['embedding_size']])

for i in range(0, len(dfm_params['deep_layers'])):
    y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])
    y_deep = tf.nn.relu(y_deep)

"""final layer"""
if dfm_params['use_fm'] and dfm_params['use_deep']:
    concat_input = tf.concat([fm_first_order, fm_second_order, y_deep], axis=1)
elif dfm_params['use_fm']:
    concat_input = tf.concat([fm_first_order, fm_second_order], axis=1)
elif dfm_params['use_deep']:
    concat_input = y_deep

# item_index = tf.slice(feat_index,[0,0],[label.shape], name='item_index')
out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias']) )
outj = tf.add(tf.matmul(tf.concat([fm_first_order, fm_second_order, y_deep], axis=1), weights['concat_projection']), weights['concat_bias'],name="add_out")
outpre = tf.concat([outj, label],1)

"""loss and optimizer"""
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(dfm_params['learning_rate'], global_step,dfm_params['learn_rate_step'], dfm_params['decay'], staircase=True)
loss = tf.losses.mean_squared_error( tf.reshape(label, (-1, 1)),outj) # epoch 99,loss is 3.3049653
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss,global_step=global_step)
# optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'], beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss)

# feature_index, feature_value, lable, dfm_param = get_data(dfm_params)
i_batch, v_batch, label_batch = get_batch(train_feature_index,train_feature_value,train_y, dfm_params['batch_size'])

"""train"""
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
config=tf.ConfigProto(gpu_options=gpu_options)
def getAuc(predict_playNum):
    p=[]
    n=[]
    count=0
    for i in predict_playNum.index:
        row = predict_playNum.iloc[i, :]
        if row[1]>=4:
            p.append(row[0])
        else:
            n.append(row[0])
    for rowp in p:
        for rown in n:
            if rowp>rown:
                count=count+1
    return count / (len(p)*len(n))

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("D:\\Anaconda3\\Scripts\\logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # tf.train.Saver().restore(sess, save_path=model_path)
    for i in range(30):
        epoch_loss, _ = sess.run([loss, optimizer], feed_dict={feat_index: train_feature_index,
                                                               feat_value: train_feature_value,
                                                               label: train_y})
        print("epoch %s,loss is %s" % (str(i), str(epoch_loss)))
    tf.train.Saver().save(sess, save_path=model_path)



        # 测试
    tf.train.Saver().restore(sess, save_path=model_path)
    # epoch_loss = sess.run([loss, optimizer], feed_dict={feat_index: train_feature_index,  feat_value: train_feature_value,  label: train_y})
    predict_label = sess.run([outpre], feed_dict={feat_value: train_feature_value, feat_index: train_feature_index, label: train_y})
    recDf = pd.DataFrame(predict_label[0], columns=['score', 'playNum'])
    print(getAuc(recDf))

   

