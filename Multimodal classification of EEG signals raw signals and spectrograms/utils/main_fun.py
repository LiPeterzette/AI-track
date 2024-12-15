import datetime
import os
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from preprocess import preprocess
from my_splite import data_split
from utils.device_to_use import usable_gpu

# pytorch
# 测试编号
time_len = 1
test_version = 'AGCN18A3'
dataset_name = 'KUL'

# 模型设置
graph_layer_num = 1
graph_convolution_kernel = 10

l_freq, h_freq = 12, 30
is_channel_attention = True

lr = 1e-3
epochs = 300
overlap = 0
is_ica = True
is_cross_subject = False
subject_num_dict = {'KUL': 16, 'DTU': 18, 'SCUT': 20}
subject_num = subject_num_dict[dataset_name]
sample_len,  channels_num = int(128 * time_len), 64

result_folder_name = f'V{test_version}_{dataset_name}_{time_len}s_e{epochs}_' \
                     f'{l_freq}to{h_freq}Hz_' \
                     f'ica{str(is_ica)[0]}_' \
                     f'ca{str(is_channel_attention)[0]}_' \
                     f't{datetime.datetime.now().strftime("%d%H%M")}'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭tensorflow的输出
edges = np.load('../bin/utils/edges.npy')
edges = edges.transpose(1, 0)
edges1 = np.copy(edges)
edges1[[0,1],:]= edges[[1,0],:]

# 转换成无向图
edges = np.concatenate((edges1, edges), axis=1)
num_edges = len(edges[1])

batch_size = 1

# class MyChannelAttention(keras.layers.Layer):
#     def __init__(self):
#         super(MyChannelAttention, self).__init__()
#         self.channel_attention = keras.models.Sequential([
#             keras.layers.GlobalAvgPool2D(),
#             keras.layers.Dense(4, activation='tanh'),
#             keras.layers.Dense(channels_num),
#         ])
#
#     def build(self, input_shape):
#         super(MyChannelAttention, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         inputs = tf.transpose(inputs, (0, 1, 3, 2))
#         cha_attention = self.channel_attention(inputs)
#
#         cha_attention = tf.reduce_mean(cha_attention, axis=0)
#
#         return cha_attention
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#
# class MyGraphConvolution(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(MyGraphConvolution, self).__init__(**kwargs)
#         # 导入邻接矩阵
#         adjacency = np.zeros((64, 64))  # solve the adjacency matrix (N*N, eg. 64*64)
#         # edges = np.load('../bin/utils/edges.npy')
#         for x, y in edges:
#             adjacency[x][y] = 1
#             adjacency[y][x] = 1
#         adjacency = np.sign(adjacency + np.eye(channels_num))
#         #度矩阵计算
#         adjacency = np.sum(adjacency, axis=0) * np.eye(64) - adjacency
#         #计算特征值与特征向量
#         e_vales, e_vets = np.linalg.eig(adjacency)
#
#
#
#         # 计算模型需要的参数
#         self.adj = None
#         self.e_vales = tf.cast(e_vales, dtype=tf.float32)
#         self.e_vets = tf.cast(e_vets, dtype=tf.float32)
#
#         # graph_convolution_kernel=10
#         # 计算 图卷积 的卷积核
#         graph_kernel = self.add_weight(shape=[graph_convolution_kernel, 1, channels_num])
#
#         graph_kernel = graph_kernel * tf.eye(channels_num)
#         # graph_kernel:10*64*64
#         graph_kernel = tf.matmul(tf.matmul(self.e_vets, graph_kernel), tf.transpose(self.e_vets, (1, 0)))
#         #
#         self.graph_kernel = tf.expand_dims(graph_kernel, axis=0)
#
#         # plt.matshow(self.graph_kernel[0,0])
#
#         # 添加 注意力 机制
#         self.graph_channel_attention = MyChannelAttention() if is_channel_attention else []
#
#     def build(self, input_shape, **kwargs):
#         super(MyGraphConvolution, self).build(input_shape)  # 一定要在最后调用它
#
#     # x：batch * k * channels * times, 16 * 1||5 * 64 * 128
#     def call(self, x, **kwargs):
#        #adj为卷积核
#         adj = self.graph_kernel
#
#         # 通道注意力网络
#         if is_channel_attention:
#             cha_attention = self.graph_channel_attention(x)
#
#             adj = cha_attention * adj
#
#             # 卷积过程
#         x = tf.matmul(adj, x)
#         x = keras.layers.Activation('relu')(x)
#
#         return x
#
#     @staticmethod
#     def compute_output_shape(input_shape):
#         return input_shape
#
# class SAGPoolModel(tf.keras.Model):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.gcns = []
#         self.sag_pools = []
#         self.gcns.append(GCN(128, activation=tf.nn.relu))
#         self.sag_pools.append(SAGPool(
#             score_gnn=GCN(1),
#             ratio=0.5,
#             score_activation=tf.nn.tanh
#         ))
#         self.mlp = tf.keras.Sequential([
#             tf.keras.layers.Permute((1, 3, 2)),
#             tf.keras.layers.AvgPool2D((1, sample_len)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(8, activation='tanh'),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(2, activation='softmax'),
#         ])
#         self.pre = tf.keras.Sequential([
#             tf.keras.layers.Permute((2, 1), input_shape=
#             (sample_len, channels_num)),
#             tf.keras.layers.Reshape((1, channels_num, sample_len)),
#             tf.keras.layers.BatchNormalization(axis=1),
#         ])
#     def call(self, inputs, training=None, mask=None, cache=None):
#         x, edge_index, edge_weight, node_graph_index = inputs
#         h = x
#         h = self.pre(h, training=training)
#         h = gcn([h, edge_index, edge_weight], training=training)
#         h, edge_index, edge_weight, node_graph_index = sag_pool([h, edge_index, edge_weight, node_graph_index],
#                                                                 training=training)
#         h = self.mlp(h, training=training)
#         return h


# class MySAGPlayer(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(MySAGPlayer, self).__init__(**kwargs)
#         self.score_gnn = GCN(1)
#         self.k = None
#         self.ratio = 0.5
#         self.score_activation = tf.nn.tanh
#
#     def build(self, input_shape, **kwargs):
#         super(MySAGPlayer, self).build(input_shape)  # 一定要在最后调用它
#     def call(self, inputs, **kwargs):
#         inputs = tf.transpose(inputs, (0, 1, 3, 2))
#         x, edge_index, edge_weight, node_graph_index = inputs
#         pooled_graph.x, pooled_graph.edge_index, pooled_graph.edge_weight, pooled_graph.node_graph_index = sag_pool(x, edge_index, edge_weight, node_graph_index,
#              self.score_gnn, self.k, self.ratio,
#              self.score_activation, training=None, cache=None)
#
#         return

class MyGraphConvolution(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyGraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape, **kwargs):

        super(MyGraphConvolution, self).build(input_shape)  # 一定要在最后调用它

    # x：batch * k * channels * times, 16 * 1||5 * 64 * 128
    def call(self, inputs, **kwargs):
        x, edges = inputs
        edges_newnum = len(edges[1])
        adjacency = np.zeros((edges_newnum, edges_newnum))
        for x, y in edges:
            adjacency[x][y] = 1
            adjacency[y][x] = 1
        adjacency = np.sign(adjacency + np.eye(channels_num))
        #度矩阵计算
        adjacency = np.sum(adjacency, axis=0) * np.eye(64) - adjacency
        #计算特征值与特征向量
        e_vales, e_vets = np.linalg.eig(adjacency)
        # 计算模型需要的参数
        adj = None
        e_vales = tf.cast(e_vales, dtype=tf.float32)
        e_vets = tf.cast(e_vets, dtype=tf.float32)

        # graph_convolution_kernel=10
        # 计算 图卷积 的卷积核
        graph_kernel = self.add_weight(shape=[graph_convolution_kernel, 1, channels_num])
        graph_kernel = graph_kernel * tf.eye(channels_num)
        # graph_kernel:10*64*64
        graph_kernel = tf.matmul(tf.matmul(e_vets, graph_kernel), tf.transpose(e_vets, (1, 0)))
        self.graph_kernel = tf.expand_dims(graph_kernel, axis=0)

       #adj为卷积核
        adj = self.graph_kernel

        # 通道注意力网络
        # if is_channel_attention:
        #     cha_attention = self.graph_channel_attention(x)
        #
        #     adj = cha_attention * adj

            # 卷积过程
        x = tf.matmul(adj, x)
        x = keras.layers.Activation('relu')(x)

        return x

















# def create_model():
#     # set the model
#     model = Sequential()
#     # the input data preprocess
#     model.add(keras.layers.Permute((2, 1), input_shape=
#     (sample_len, channels_num)))
#     model.add(keras.layers.Reshape((1, channels_num, sample_len)))
#     model.add(keras.layers.BatchNormalization(axis=1))
#
#     # convolution module
#     for k in range(graph_layer_num):
#         model.add(MyGraphConvolution())
#         model.add(keras.layers.BatchNormalization(axis=1))
#
#     # 全连接分类器
#     model.add(keras.layers.Permute((1, 3, 2)))
#     model.add(keras.layers.AvgP
#     pool2D((1, sample_len)))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.Dense(8, activation='tanh'))
#     model.add(keras.layers.Dropout(0.3))
#     model.add(keras.layers.Dense(2, activation='softmax'))
#
#
#     # set the optimizers
#     model.compile(
#         optimizer=optimizers.Adam(lr),
#         loss=losses.BinaryCrossentropy(),
#         metrics=['accuracy'],
#     )
#
#     return model




def main(sub_id='1'):
    # 打印控制参数
    sub_id = str(sub_id)
    print(f'sub_id={sub_id}')
    print(result_folder_name)
    print()

    # 设定使用的GPU
    if not is_debug:
        gpu = usable_gpu[int(sub_id) % len(usable_gpu)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


    # 设置回调函数
    epoch_early_stop = 10 if is_cross_subject else 50
    callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                                           patience=epoch_early_stop, min_delta=1e-3)

    # 加载数据并训练
    # TODO：合并模型，通过保存数据的索引，变成一个列表，就可以实现训练过程统一化。
    if is_cross_subject:
        # data loader: trail * time * channels, fs = 128Hz
        x_train = np.zeros((0, sample_len, channels_num))
        y_train = np.zeros((0, 2))

        # 加载所有人的数据
        for k_sub in range(subject_num):
            print('sub:' + str(k_sub))

            eeg, voice, label = preprocess(dataset_name, sub_id, l_freq=l_freq, h_freq=h_freq, is_ica=True)
            eeg, voice, label = data_split(eeg, voice, label, 1, time_len, overlap)
            data = eeg
            # eeg:样本数*样本长度128*通道数
            if int(k_sub) == int(sub_id) - 1:
                x_test, y_test = data, label
            else:
                x_train = np.concatenate((x_train, data), axis=0)
                y_train = np.concatenate((y_train, label), axis=0)

        # 打乱数据（训练集）
        index = [i for i in range(x_train.shape[0])]
        random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        # 打乱数据（测试集）
        index = [i for i in range(x_test.shape[0])]
        random.shuffle(index)
        x_test = x_test[index]
        y_test = y_test[index]
        print(x_train.shape)
        print(y_train.shape)

        # 训练模型

        model = create_model()
        callbacks = [callback_early_stop]
        model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=callbacks, verbose=False)

        # 测试模型
        loss, acc = model.evaluate(x_test, y_test)
        del model

        accuracies = [acc, acc, acc, acc, acc]
        print([acc])
        print()
    else:
        # data loader: trail * time * channels, fs = 128Hz
        eeg, voice, label = preprocess(dataset_name, sub_id, l_freq=l_freq, h_freq=h_freq, is_ica=True)
        eeg, voice, label = data_split(eeg, voice, label, 1, time_len, overlap)


        label = tf.one_hot(label, 2).numpy()
        data = eeg
        data = data.transpose(0, 2, 1)
        # edges = np.expand_dims(edges, axis=0)

        # 训练模型
        accuracies = []
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(data, label):
            # 打乱索引顺序
            np.random.shuffle(train_index)
            np.random.shuffle(test_index)

            # 数据划分
            x_tra, y_tra = data[train_index], label[train_index]
            x_test, y_test = data[test_index], label[test_index]

            sample_tranum = len(x_tra[0])
            sample_testnum = len(x_test[0])

            weight_tra = np.ones((sample_tranum, num_edges))



            edges_tra = edges
            edges_test = edges
            for i in range(sample_tranum):
                edges_tra = np.concatenate((edges_tra, edges), axis=0)
            for i in range(sample_testnum):
                edges_test = np.concatenate((edges_test, edges), axis=0)
            label_tra = [np.argmax(one_hot) for one_hot in y_tra]
            label_test = [np.argmax(one_hot) for one_hot in y_test]
            label_tra = np.expand_dims(label_tra, axis=1)


            # loss 可视化

            callbacks = [callback_early_stop]  # 不可以移到五折之外，会导致模型性能发生变化。
            if is_debug:
                log_dir = f'../result/debug/S{sub_id}/{result_folder_name}_{test_index[0]}'
                callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))


            # 训练模型

            model = multi_input_model()

            def forward(batch_graph, training=False):
                return model(
                    [batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight, batch_graph.node_graph_index],
                    training=training)

            def evaluate(test_graphs):
                accuracy_m = keras.metrics.Accuracy()

                for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
                    logits = forward(test_batch_graph)
                    preds = tf.argmax(logits, axis=-1)
                    accuracy_m.update_state(test_batch_graph.y, preds)

                return accuracy_m.result().numpy()

            # model.compile(
            #     optimizer=optimizers.Adam(lr),
            #     loss=losses.BinaryCrossentropy(),
            #     metrics=['accuracy'],)
            # label_tra = np.expand_dims(label_tra, axis=1)
            # label_tra1 = label_tra
            # for i in range(64):
            #     label_tra1 = np.concatenate((label_tra1, label_tra), axis=1)
            #
            # model.fit([x_tra, edges_tra, weight, label_tra1], [y_tra], epochs=epochs, callbacks=callbacks, verbose=False)



            # 测试模型
            # sample_testnum = len(x_tra[0])
            # label_input = keras.Input(shape=(sample_testnum))



            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

            train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

            for step in tqdm(range(300)):
                train_batch_graph = next(train_batch_generator)
                with tf.GradientTape() as tape:
                    logits = forward(train_batch_graph, training=True)
                    losses = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
                    )

                vars = tape.watched_variables()
                grads = tape.gradient(losses, vars)
                optimizer.apply_gradients(zip(grads, vars))

                if step % 20 == 0:
                    mean_loss = tf.reduce_mean(losses)
                    accuracy = evaluate(test_graphs)
                    print("step = {}\tloss = {}\taccuracy = {}".format(step, mean_loss, accuracy))

            loss, acc = model.evaluate([x_test, edges_test, weight, label_test], [y_test])
            accuracies.append(acc)
            model.summary()
            del model

            print(accuracies)
            print()

    # 打印结果并输出
    print(result_folder_name)
    print(np.mean(accuracies))
    print(accuracies)

    return accuracies



if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        is_debug = False
        main(sys.argv[1])
    else:
        is_debug = True
        main()
