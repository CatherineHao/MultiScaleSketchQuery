import time
import os
from tkinter import LEFT
from traceback import print_tb
from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Lambda, Layer
from keras.optimizers import Adam, RMSprop, Adadelta
import keras.backend as K
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
import torch

def GetTrainData(TrainDataFile):
    # 时序数据训练集
    TimeSeriesTrainData = []
    # 用户草图训练集
    SkTrainData = []
    # 相似度数据
    SimData = []
    # 用户评分
    df = pd.read_csv(TrainDataFile)
    # global X_train,Y_train,df,max_seq_length
    for rowId, row in df.iterrows():
        # 生成sketch数据
        tmp = row["Sketch"].split(";")
        TempSketch = []
        for i in range(0,len(tmp)):
            # 将str转化为对应的array
            t = tmp[i].replace('(','')
            t = t.replace(')','')
            t = t.strip()
            t = t.strip("\n")
            t = t.strip("\t")
            # 判断是否有空字符 print(t.isspace())  print('tmp:',t)
            tmparr = t.split(',')
            #print('1:',float(tmparr[0]))
            #print('2:',float(tmparr[1]))
            arraytmp = np.array([float(tmparr[0]),float(tmparr[1])])
            #print(arraytmp)
            #print('shepe:',arraytmp.shape)
            #print(np.array(tmp[i]).shape) #()
            #TempSketch.append(float(np.array(tmp[i])))
            TempSketch.append(arraytmp)
        SkTrainData.append(TempSketch)
        #print('SkTrainData Type:', type(SkTrainData))
        
        # 生成时间数据
        tmp = row["MatchSeries"].split(";")
        TempTimeSeries = []
        for i in range(0,len(tmp)):
            TempTimeSeries.append(float(tmp[i]))
            #TempTimeSeries.append(tmp[i])
        TimeSeriesTrainData.append(TempTimeSeries)
        # print('TimeData Type:', type(TimeSeriesTrainData)) # type= list

        # 取出相似度
        SimData.append(float(row["Sim"]))
    print("SketchTrainData Length: ", len(SkTrainData))
    print("TimeSeriesTrainData Length: ", len(TimeSeriesTrainData))
    print("SimData Length: ", len(SimData))
    return SkTrainData, TimeSeriesTrainData, SimData

def TrainDataPreprocess(SketchTrainData, TimeSeriesTrainData):
    NewSketchData = []
    for i in range(len(SketchTrainData)):
        tmpSketchData = []
        for j in range(1, len(SketchTrainData[i])):
            tmpSketchData.append(SketchTrainData[i][j] - SketchTrainData[i][j-1])
        NewSketchData.append(tmpSketchData)


    NewTimeSeriesData = []
    for i in range(len(TimeSeriesTrainData)):
        tmpTimeSeriesData = []
        for j in range(1, len(TimeSeriesTrainData[i])):
            tmpTimeSeriesData.append(TimeSeriesTrainData[i][j] - TimeSeriesTrainData[i][j-1])
        NewTimeSeriesData.append(tmpTimeSeriesData)
    # 测试
    return NewSketchData, NewTimeSeriesData

def TrainDataPreprocessZScore(SketchTrainData, TimeSeriesTrainData):
    NewSketchData = []
    for i in range(len(SketchTrainData)):
        NewSketchData.append(zscore(SketchTrainData[i]))
    NewTimeSeriesData = []
    for i in range(len(TimeSeriesTrainData)):
        NewTimeSeriesData.append(zscore(TimeSeriesTrainData[i]))
    # 测试
    return NewSketchData, NewTimeSeriesData


def exponent_neg_manhattan_distance(left, right):
    '''Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=-1, keepdims=True))


def euclidean_distance(left, right):
    return K.sqrt(K.sum(K.square(left - right), axis=-1, keepdims=True))


def euclidean_distance_output_shape(shape):
    shape1, shape2 = shape
    return (shape1[0] , 1)

class SaveWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            self.model.save("./config/FINALLSTMV3-{}-.h5".format(epoch))
            
# 创建contrastive_loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)            

def TrainNetwork(SketchData, TimeSeriesData, SimData, FileName):
    #batch_size = len(SketchData)
    batch_size = 32
    hunits = 16 # 单元内部隐藏层的大小
    n_epoch = 10 # epoch个数
    gradient_clipping_norm = 1.05
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)是keras的一种优化器，以在训练初期加快训练
    # amsgrad: 布尔型，是否使用AMSGrad变体
    adam = Adam(learning_rate=0.001)
    
    shared_model = LSTM(hunits)
    
    left_input = Input(shape=(len(TimeSeriesData[0]),2), dtype='float')
    #right_input = Input(shape=(len(SketchData[0]),1), dtype='float')
    right_input = Input(shape=(len(SketchData[0]),2), dtype='float')
    left_output = shared_model(left_input)
    right_output = shared_model(right_input)
    malstm_distance = Lambda(function=lambda x: euclidean_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    # 定义输入和输出
    # model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    model = Model(inputs=[left_input, right_input], outputs=malstm_distance)
    # Adadelta optimizer, with gradient clipping by norm
    
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # 评价函数用于评估当前训练模型的性能，当模型编译后（compile），评价函数应该作为metrics的参数来输入。
    # 评价函数和损失函数相似，只不过评价函数的结果不会用于训练过程中。我们可以传递已有的评价函数名称，或者传递一个自定义的 Theano/TensorFlow 函数来使用。
    # optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    # rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy']) # 损失函数为均方误差
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    training_start_time = time.time()
    # malstm_trained = model.fit([np.array(TimeSeriesData).reshape(-1, len(TimeSeriesData[0]), 1), np.array(SketchData).reshape(-1, len(SketchData[0]), 1)], np.array(SimData), batch_size=batch_size, epochs=n_epoch,validation_split=0.3, callbacks=[MySaver])
    TimeUse = torch.tensor(np.array(TimeSeriesData).reshape(-1, len(TimeSeriesData[0]), 1))
    TimeUse = TimeUse.expand(14100,len(TimeSeriesData[0]),2)
    TimeUse = TimeUse.numpy()
    malstm_trained = model.fit([TimeUse, np.array(SketchData).reshape(-1,len(SketchTrainData[0]),2)], np.array(SimData), batch_size=batch_size, epochs=n_epoch,validation_split=0.2)
    
    training_end_time = time.time()

    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))
    model.save('./config/' + FileName)
    
    # plot accuracy
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
    print("Done.")

    
path = "../TrainData1028" # 训练数据文件夹
files = os.listdir(path) 
# 时序数据训练集
TimeTrainData = []
# 用户草图训练集
SketchTrainData = []
# 相似度数据
SimTrainData = []
for file in files:
    position = path + '\\' + file
    TmpSketch = []
    TmpTime = []
    TmpSim = []
    TmpSketch, TmpTime,TmpSim = GetTrainData(position)
    SketchTrainData.append(TmpSketch)
    TimeTrainData.append(TmpTime)
    SimTrainData.append(TmpSim)
# SketchTrainData = all_length_dataset
H5FileName = "FinalLSTMV1.h5"
#TrainNetwork(SketchTrainData, TimeTrainData, SimTrainData, H5FileName)
print(len(SimTrainData))

TrainDataFile = "../TrainDataSet/TrainDada.csv"
SketchTrainData, TimeSeriesTrainData, SimData = GetTrainData(TrainDataFile)
H5FileName = "FinalLSTM_V3.h5"
TrainNetwork(SketchTrainData, TimeSeriesTrainData, SimData, H5FileName)