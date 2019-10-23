import pandas as pd
import tensorflow as tf
import numpy as np
pd.set_option('display.max_columns', 100)  # 设置最大显示列数

TRAIN_FILE = "F:\\python\\data\\cs2\\part-00000-0920a0cb-4e95-49d4-9ddb-4d643095ffa9-c000.csv"
model_path="F:/python/model/deepFM2J2/"
rec_path="F:/python/data/predict/"
fdict_path = "F:/python/data/feature_dict3"
# TRAIN_FILE = "/mnt/data/part-00000-ca88c467-c550-43ec-9341-090c14a0dfca-c000.csv"
# model_path = "/mnt/model/"
# fdict_path = "/mnt/data/feature_dict"
# rec_path="/mnt/data/"

NUMERIC_COLS = ["penalty"]
IGNORE_COLS = ["playNum", "clickNum","disNum","collectNum","rating","target","day"]
topN=1000
"""模型参数"""
dfm_params = {
    "decay": 0.83,
    "learn_rate_step":9850,  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE
    "threads":6,
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 10,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 15,
    "batch_size": 1024,
    "learning_rate": 0.05,
    # "learning_rate":0.001,  0.1-1 0.01-10 0.002-10 0.001-10(然loss从2变3了，似乎是个别异常数据拉高的)
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": 'gini_norm',
    "random_seed": 3
}

dfTrain = pd.read_csv(TRAIN_FILE)
# dfTrain = pd.read_csv(TRAIN_FILE,nrows=1000000)#.tail(10000)
print(dfTrain.head(5)[['uId','itemID',"playNum","rating"]])
print(dfTrain.tail(5)[['uId','itemID',"playNum","rating"]])

data_rows=len(dfTrain)
item_df=dfTrain[['itemID', "penalty", "genre","artistid"]].drop_duplicates().reset_index(drop=True)
user_df=dfTrain[['uId', "country"]].drop_duplicates().reset_index(drop=True)

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
# print(feature_dict['artistid'][10002])
# print(feature_dict['artistid'][234815])
# print(feature_dict['uId'][102079130])
# with open(fdict_path, 'w+') as f:
#                 f.write(str(feature_dict) )
# with open(fdict_path, 'r') as f:
#     print(dict(f.read()).keys() )

def norm_itemID(feat_dict,item_df):
    f_index=[]
    f_value=[]
    for i in item_df.index:
        item=item_df.iloc[i,:]
        itemID=item[0]
        penalty=item[1]
        genre = item[2]
        artistid = item[3]
        f_value.append([1, 1, 1, penalty, 1, 1])
        f_index.append( [ feat_dict['itemID'][itemID], 0,0, feat_dict['penalty'], feat_dict['genre'][genre], feat_dict['artistid'][artistid] ] )
    return f_index,f_value

def norm_f_index(feat_dict, uid_country, f_index):
    #  itemID  country  uId   penalty  genre  artistid
    uid=uid_country[0]
    country=uid_country[1]
    user_index=feat_dict['uId'][uid]
    country_index=feat_dict['country'][country]
    # 要是能直接改两列的值为用户信息就好了
    for v in f_index:
        v[1]=country_index
        v[2]=user_index
    return f_index

# f_index,f_value=norm_itemID(feature_dict,  item_df)

"""
对训练集进行转化
"""
# print(dfTrain.columns)
item_i = dfTrain[['itemID']].values.tolist()
# train_y = dfTrain[["rating"]].values.tolist()
train_y = dfTrain[['playNum']].values.tolist()
train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()

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


# def get_batch(train_feature_index,train_feature_value,train_y, batch_size):
#     print(len(train_feature_index),len(train_y))
#     input_queue = tf.train.slice_input_producer([train_feature_index,train_feature_value,train_y],num_epochs=1 , shuffle=False )
#     i_batch, v_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
#     print(v_batch)
#     return i_batch, v_batch,label_batch
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
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("D:\\Anaconda3\\Scripts\\logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 开启协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    batch= int(data_rows / dfm_params['batch_size'])
    print(data_rows , dfm_params['batch_size'],batch)
    batch_x_epochs = 0 # global_step
    end_loss=0
    tf.train.Saver().restore(sess, save_path=model_path)

    try:
        while not coord.should_stop(): # batch= int(data_rows / dfm_params['batch_size']) 9000
            # 获取训练用的每一个batch中batch_size个样本和标签
            # i, v,l = sess.run([i_batch,  v_batch, label_batch])
            # print(i.shape)
            # batch_loss=1
            # for j in range(200):
            #     if j<200 and batch_loss>0.01:
            #         print(i)
            #         sess.run(optimizer, feed_dict={feat_index: i,feat_value: v, label: l})
            #         train_loss = loss.eval({feat_index: i,feat_value: v, label: l})
            #         batch_loss=train_loss
            #         if batch%10==0 and (j==1 or j==199 or batch_loss<0.01):
            #             print("batch %d,  Training loss %g" % (batch,  train_loss))
            #         if batch_loss<0.01:
            #             batch_loss = 1
            #             break
            # batch = batch + 1

            i, v, l = sess.run([i_batch, v_batch, label_batch])
            # print(i.shape)
            sess.run(optimizer, feed_dict={feat_index: i, feat_value: v, label: l})
            train_loss = loss.eval({feat_index: i, feat_value: v, label: l})
            end_loss=train_loss
            if batch_x_epochs % 400 == 0:
            # if batch_x_epochs % batch == 0:
                learning_rate_val = sess.run(learning_rate)
                global_step_val = sess.run(global_step)
                print("batch_x_epochs %d,  Training loss %g,  global_step %g,  learning_rate %g" % (batch_x_epochs, train_loss, global_step_val,learning_rate_val))
                tf.train.Saver().save(sess, save_path=model_path)
                frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    ["feat_index", "feat_value", "add_out"])
                # 保存图为pb文件
                with open(model_path + 'model.pb', 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())
            batch_x_epochs = batch_x_epochs + 1

    except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
        print("---Train end---", batch_x_epochs, end_loss)
        tf.train.Saver().save(sess, save_path=model_path)
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('---Programm end---')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束

    # 测试
    tf.train.Saver().restore(sess, save_path=model_path)
    i,predict_data = sess.run([feat_index, outj],
                              # feed_dict={feat_value: [[1, 1, 1, 0.2, 1, 1],    [1, 1, 1, 1, 1, 1],    [1, 1, 1, 0.721348, 1, 1],    [1, 1, 1, 1, 1, 1],    [1, 1, 1, 0.225091, 1, 1],
                              #                         [1, 1, 1, 0.352956, 1, 1], [1, 1, 1, 0.910239, 1, 1],                     [1, 1, 1, 0.265873, 1, 1], [1, 1, 1, 0.221151, 1, 1],                  [1, 1, 1, 0.389871, 1, 1]                   ],
                              #            feat_index: [[7389, 1476, 1439, 7388, 1444,0],    [7389, 1477, 1439, 7388, 1445,1],    [7389, 1478, 1439, 7388, 1445,2],    [7389, 1479, 1439, 7388, 1446,3],    [7389, 1480, 1439, 7388, 1447,4],
                              #                         [7441, 4741, 1439, 7388, 1450, 55],    [7441, 7387, 1439, 7388, 1450, 36],    [7441, 4973, 1439, 7388, 1452, 570],    [7441, 2479, 1439, 7388, 1444, 36],    [7441, 6560, 1439, 7388, 1444, 202]]}
                            feed_dict = {
                                feat_value: [[1, 1,1,1,1,1,1,1, 1,1], [1, 1, 1, 1, 1, 1,1,1,1, 1],   [1, 1, 1, 1, 1, 1,1,1,1, 1], [1, 1, 1, 1, 1, 1,1,1,1, 1],  [1, 1, 1, 1, 1, 1,1,1,1, 1]],
                                feat_index: [
                                    [37799, 59510,555656,555660,555766,556125,558830, 558952,558996,558998],
                                    [6527, 96509,555657,555672,555766,555767,558827, 558951,558996,559003],
                                    [1910,141989,555658,555674,555766,556068,558819, 558949,558995,559003],
                                    [1925,141989,555658,555660,555766,556192,558819, 558949,558995,559003],
                                    [14238,141990,555659,555660,555766,555988,558846, 558955,558995,559003]]
                                         }
                              )
    print("默认格式模型predict_data",predict_data)

    # 显示图中的节点
    # print([n.name for n in sess.graph.as_graph_def().node])
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ["feat_index", "feat_value",  "add_out"])

    # 保存图为pb文件
    with open(model_path + 'model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    # 推荐
    # f_index, f_value = norm_itemID(feature_dict, item_df)
    # for i in user_df.index:
    #     u = user_df.iloc[i, :]
    #     # if (u[0]%500==0):
    #     print(u[0])
    #     f_index = norm_f_index(feature_dict, u, f_index)
    #     predict_score = sess.run([ outj], feed_dict={feat_value: f_value,  feat_index: f_index}      )
    #     rec_dfi=pd.concat([item_df['itemID'], pd.DataFrame(predict_score[0],columns=['score'])], axis=1 )            .sort_values('score',ascending=False).head(topN)            .assign(uid = u[0])
    #     if (i==0):
    #         rec_df=rec_dfi
    #     else:
    #         rec_df=pd.concat([rec_df,rec_dfi], axis=0)
    # rec_df.to_csv(rec_path+'dfmRec', index=False, header=True)

