import datetime
import os
import sys
from my_splite import data_split
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from keras import Sequential, optimizers, losses
import random
from preprocess import preprocess
from my_process import process

# 测试编号
time_len = 1
test_version = 'GCN29A1'
dataset_name = 'KUL'
overlap = 0
# 模型设置
is_band_attention = True
is_channel_attention = True
# l_freq = [1, 4, 8, 12, 30]
# h_freq = [4, 8, 12, 30, 50]
l_freq = 1
h_freq = 4
lr = 1e-3
epochs = 100
is_ica = True
is_cross_subject =False
subject_num_dict = {'KUL': 16, 'DTU': 18, 'SCUT': 20}
subject_num = subject_num_dict[dataset_name]
sample_len, channels_num = int(128 * time_len), 64

batch_dirt = {0.1: 256, 0.2: 256, 0.5: 64, 1: 32, 2: 16, 5: 8, 10: 4}
batch_size = batch_dirt[time_len] if not is_cross_subject else 512

result_folder_name = f'V{test_version}_{dataset_name}_{time_len}s_e{epochs}_' \
                     f'ica{str(is_ica)[0]}_' \
                     f'ca{str(is_channel_attention)[0]}_' \
                     f't{datetime.datetime.now().strftime("%d%H%M")}'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭tensorflow的输出

if channels_num == 64:
    channel_index = []
elif channels_num == 32:
    channel_index = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24, 27, 29, 31, 32, 34, 36, 38, 40,
                     42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 61]
elif channels_num == 16:
    channel_index = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 29, 31, 32, 34, 35,
                     36, 38, 40, 41, 42, 43, 44, 45, 46, 48, 50, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62]
channel_index = set(range(64)) - set(channel_index)
channel_index = list(channel_index)


def inception_resnet_stem():
    inputs=keras.layers.Input((sample_len, channels_num))

    input=keras.layers.Reshape((sample_len, channels_num, 1), input_shape=[sample_len, channels_num])(inputs)
    input=keras.layers.BatchNormalization(-1)(input)
    conv1=keras.layers.Conv2D(5, (16,64),activation='relu')(input)

    conv1=keras.layers.AveragePooling2D(pool_size=(112,1))(conv1)

    out=keras.layers.Flatten()(conv1)
    # out=keras.layers.Dropout(0.5)(out)
    out=keras.layers.Dense(5, activation='sigmoid')(out)
    # out=keras.layers.Dropout(0.5)(out)
    outs=keras.layers.Dense(2, activation='softmax')(out)

    model=keras.Model(inputs=[inputs],outputs=[outs])
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model

def create_model():
    inputs=keras.layers.Input((sample_len, channels_num))

    input=keras.layers.Reshape((sample_len, channels_num, 1), input_shape=[sample_len, channels_num])(inputs)
    input=keras.layers.BatchNormalization(-1)(input)
    conv1=keras.layers.Conv2D(5, (17,64),activation='relu')(input)

    conv1=keras.layers.AveragePooling2D(pool_size=(112,1))(conv1)

    out=keras.layers.Flatten()(conv1)
    # out=keras.layers.Dropout(0.5)(out)
    out=keras.layers.Dense(5, activation='sigmoid')(out)
    # out=keras.layers.Dropout(0.5)(out)
    outs=keras.layers.Dense(2, activation='softmax')(out)

    model=keras.Model(inputs=[inputs],outputs=[outs])
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def main(sub_id):
  sub=[]
  Acc=[]
  # 打印控制参数
  for sub_id in range(1, 17):
      sub_id = str(sub_id)
      print(f'Test sub_id={sub_id}')
      # print(result_folder_name)
      print()



      # 设置回调函数
      epoch_early_stop = 10 if is_cross_subject else 100
      callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                                             patience=epoch_early_stop, min_delta=1e-3)

      lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=30, verbose=1,
                                                         mode='max', min_delta=1e-6, cooldown=0, min_lr=1e-6)

      # 加载数据并训练
      # TODO：合并模型，通过保存数据的索引，变成一个列表，就可以实现训练过程统一化。
      if is_cross_subject:
          # data loader: trail * time * channels, fs = 128Hz
          x_train = np.zeros((0, sample_len, channels_num))
          y_train = np.zeros((0, 2))

          # 加载所有人的数据
          for k_sub in range(subject_num):
              # print('sub:' + str(k_sub))
              data, label = preprocess(dataset_name, sub_id, l_freq, h_freq, is_ica, time_len)
              # data, label = preprocess(k_sub, time_len)
              if int(k_sub) == int(sub_id) - 1:  # 测试sub_id
                  x_test, y_test = data, label
              else:
                  x_train = np.concatenate((x_train, data), axis=0)  # 将其他人的作为训练集
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
          print('have load data')
          # 训练模型
          model = create_model()
          callbacks = [callback_early_stop, lr_callback]
          model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=callbacks, verbose=2,
                    batch_size=32)

          # 测试模型
          loss, acc = model.evaluate(x_test, y_test)
          del model

          # accuracies = [acc, acc, acc, acc, acc]
          print([acc])
          print()
      else:

          eeg, voice, label = preprocess(dataset_name, sub_id, l_freq=l_freq, h_freq=h_freq, is_ica=True)
          eeg, voice, label = data_split(eeg, voice, label, 1, time_len, overlap)
          data = eeg
          label = tf.one_hot(label, 2).numpy()
          # data, label = process(sub_id, time_len)
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

              # loss 可视化

              callbacks = [callback_early_stop, lr_callback]  # 不可以移到五折之外，会导致模型性能发生变化。
              if is_debug:
                  log_dir = f'../result/debug/S{sub_id}/{result_folder_name}_{test_index[0]}'
                  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

              # 训练模型
              model = create_model()
              model.summary()
              # tf.keras.backend.set_learning_phase(True)
              model.fit(x_tra, y_tra, epochs=epochs, validation_split=0.1, callbacks=callbacks, verbose=2,
                        batch_size=32)

              # 测试模型
              # tf.keras.backend.set_learning_phase(False)
              loss, acc = model.evaluate(x_test, y_test)
              accuracies.append(acc)
              del model

              print(accuracies)
              print()

      # 打印结果并输出
      print(result_folder_name)
      print('S' + sub_id + ': ', acc)


      sub.append(sub_id)
      Acc.append(np.mean(acc))
  print(sub)
  print(Acc)

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        is_debug = False
        # for i in range(1,4):
        #    main()#
    else:

        is_debug = True
        main(1)
