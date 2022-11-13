import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
from tqdm import tqdm

import utils
import LTTB
import EMA


def get_tuple(indexed_tuple, index):
    tmp1 = indexed_tuple[index][1]
    tmp2 = indexed_tuple[index][2]
    return np.array([tmp1, tmp2])

def EDDist(s1, s2): # 必须s1是针对时间序列的，s2是针对草图序列
    dist = 0
    for i in range(len(s1)):
        vec1 = get_tuple(s1, i)
        vec2 = get_tuple(s2, i)
        dist = dist + np.sqrt(np.sum(np.square(vec1-vec2)))
    return dist

# # 最终输出
# Final = []

# # 读入需要匹配的时间序列数据
# #LongTimeSeriesData = pd.read_csv('./test0919/QueryResult2016pm25.csv')
# LongTimeSeriesData = pd.read_csv('./poll_data/60288.csv')
# LongTimeSeriesDataY = LongTimeSeriesData['so2'].values.tolist()
# LenX = len(LongTimeSeriesDataY)
# TimeX = np.linspace(1,LenX,LenX)
# TimeSeries = list(zip(TimeX,LongTimeSeriesDataY))


# convert from time series tuples to derivatives
def convert_to_derivative(ts_tuple):
    derivative = []
    for i in range (len(ts_tuple) - 1):
        tangent = (ts_tuple[i+1][1] - ts_tuple[i][1]) / (ts_tuple[i+1][0] - ts_tuple[i][0])
        length = ts_tuple[i+1][0] - ts_tuple[i][0]

        derivative.append(((int(ts_tuple[i][0]), tangent, length)))
    return derivative

sketch_folder = "/home/haojianing/sketch/sketch_data" # 文件夹目录
raw_data_folder = "/home/haojianing/sketch/raw_data" # 文件夹目录
raw_data_files = ['60288.csv'] #['60287.csv', '60288.csv', '60289.csv', '60290.csv']
pollutant = ['so2', 'pm25', 'pm10', 'no2', 'co', 'o3']

raw_data = []

# 排序定义函数
def TakeLast(elem):
    return elem[1]

def read_raw_data(raw_data_files):
    for file in raw_data_files:
        file_path = os.path.join(raw_data_folder, file)
        print(file_path, pollutant[0])
        raw_ts = pd.read_csv(file_path)
        
        ts_y = raw_ts['so2'].values.tolist()
        length = len(ts_y)
        ts_x = np.linspace(1, length, length)

        raw_data = list(zip(ts_x, ts_y))
    return raw_data

def show_top_n_result(QueryResults,SketchMappings, Blanks):
    TopResults = []
    # 第一个相似度最高因此必须添加
    TopResults.append(QueryResults[0])
    # 元素要和已经找到的依次比较  每次取出的一段元素
    for i in range(1, len(QueryResults)):
        # 假设当前元素满足题意
        flag = True
        for j in range(len(TopResults)):
            if abs(QueryResults[i][0] - TopResults[j][0]) <= Blanks:  # Blanks查找间距  类似步长
                flag = False
                break
        # 满足题意保留  匹配
        if flag == True:
            # 找到了指定数量的结果
            if len(TopResults) == SketchMappings:  # 一个草图匹配sketchmapping个结果
                break
            TopResults.append(QueryResults[i])
    print("The Query results :", TopResults)
    return TopResults

def convert_to_dataset(lttb_raw_data,derivative_sketch,QueryResults,QueryScale,Sim):
    SketchTmp = ";".join([str(num) for num in derivative_sketch])
    Final_dataset = []
    for i in range(0,len(QueryResults)):
        YList = [] # YList为对应的长度为16的处理后的segementtime值
        for j in range(QueryScale):
            YList.append(lttb_raw_data[QueryResults[i][0]:QueryResults[i][0] + QueryScale][j][1])
        TimeTmp = ";".join([str(num) for num in YList]) 
        ED_dis = QueryResults[i][1]
        tmp_sim = Sim[i]
        Final_dataset.append([SketchTmp,TimeTmp,ED_dis,tmp_sim])
    # print('tmp_dataset:',Final_dataset)
    return Final_dataset

lttb_thre = 17

train_dataset = []

if __name__=="__main__":
    raw_data = read_raw_data(raw_data_files)
    train_dataset = []
    sketch_files = os.listdir(sketch_folder) # 文件夹下所有文件名称
    for sketch_file in sketch_files:
        if sketch_file.__contains__('csv'):
            csv_file = os.path.join(sketch_folder, sketch_file)
            sketch = pd.read_csv(csv_file)

            cleaned_sketch = utils.process_sketch(sketch)
            normalized_sketch = utils.normalize_sketch(cleaned_sketch)
            
            # 进行LTTB处理
            lttb_sketch = LTTB.largest_triangle_three_buckets(normalized_sketch, lttb_thre)
            # 转成导数
            derivative_sketch = convert_to_derivative(lttb_sketch)
        
            # 随机不同的 lttb threshold，来得到不同的对应时间序列数据进行匹配
            for i in range(3):
                '''需要EMA则对theta进行推算，并设置对应的threshold值
                theta = 0.3334 # 日0.3334 周0.904762 月 0.977778
                EmaTime = calculateEMA(LongTimeSeriesDataY,theta)
                Timehold = int(len(EmaTime) / 1.5) # 日对应1.5 周对应10.5 月对应45
                '''
                # time_thre = random.randint(1500, len(raw_data))
                if i % 3 == 0:
                    time_thre = int(len(raw_data) / 1.5)
                if i % 3 == 1:
                    time_thre = int(len(raw_data) / 10.5)
                if i % 3 == 2:
                    time_thre = int(len(raw_data) / 45)
                lttb_raw_data = LTTB.largest_triangle_three_buckets(raw_data, time_thre)
                derivative_raw_data = convert_to_derivative(lttb_raw_data)
            
                QueryScale = len(derivative_sketch)
                EDResults = []  # 记录查询ED结果
                SketchMappings = 100
                Blanks = 5

                # 计算欧式距离ED，得到与对应的最相似的三条子时间序列，观察是否相似
                for window in range(0, len(derivative_raw_data) - QueryScale + 1):
                    # 滑动窗口匹配数据
                    s1 = derivative_raw_data[window: window + QueryScale]
                    # ED_dis = EDdist(s1,s2) # 原始只针对y的对比
                    ED_dist = EDDist(s1, derivative_sketch)
                    EDResults.append([window, ED_dist])

                QueryResults = EDResults # 创建一个新的list进行排序
                # 按照相似性从高低排序
                QueryResults.sort(key=TakeLast)  # takeLast记录相似性  按照相似性进行排序
                MaxDis = QueryResults[-1][1]
                MinDis = QueryResults[0][1]
                # 返回最匹配的前num个结果
                QueryResults = show_top_n_result(QueryResults, SketchMappings, Blanks)
                # 计算对应的similarity
                Sim = []
                for x in range(0,len(QueryResults)):
                    # 归一化后，sim = 1-dis,sim=[0,1]
                    tmp_sim = 1 - ((QueryResults[x][1] - MinDis) / (MaxDis - MinDis))
                    Sim.append(tmp_sim)
                print('Similarity:',Sim)

                # 导出对应文件，第一列，存储查询序列的c,l,第二列，查询到的数据SegementTime对应的y值,第三列存储对应的ED，第四列对应对应计算的sim
                tmp_dataset = convert_to_dataset(lttb_raw_data,derivative_sketch,QueryResults,QueryScale,Sim)
                # 绘制匹配的结果，QueryResults按照相似性进行了排序，绘制每一个对应的Pair,草图值+匹配的原始时间序列数据.命名为对应的草图+污染物+sim值
                sketch_x = []
                sketch_y = []
                for tmp in range(len(lttb_sketch)):
                    sketch_x.append(lttb_sketch[tmp][0])
                    sketch_y.append(lttb_sketch[tmp][1])
                for item in range (0,len(QueryResults)):
                    time_plot_x = []
                    time_plot_y = []
                    for q in range(QueryScale):
                        time_plot_x.append(lttb_raw_data[QueryResults[item][0]:QueryResults[item][0] + QueryScale][q][0])
                        time_plot_y.append(lttb_raw_data[QueryResults[item][0]:QueryResults[item][0] + QueryScale][q][1])
                    minx = min(time_plot_x)
                    maxx = max(time_plot_x)
                    fig=plt.figure()
                    sketch_fig = fig.subplots()
                    tim_fig = sketch_fig.twiny()  
                    plt.xlim(minx,maxx)
                    sketch_fig.plot(sketch_x,sketch_y,color='orange')
                    tim_fig.plot(time_plot_x,time_plot_y,color='blue')
                    name = str(sketch_file)
                    plt.savefig('../../ED_query_results/pic_sketch{}_sim{}.png'.format(name,Sim[item]))
                    plt.clf()
                train_dataset.append(tmp_dataset)
    #print('train_dataset:',train_dataset)

# 数据集构建完成，最后一步，导出为csv文件
name=['Sketch','MatchSeries','ED_dis','Sim']
train_dataset_file = pd.DataFrame(columns=name,data=train_dataset)
#len(TestData)
train_dataset_file.to_csv('./TrainDataSet.csv',index=False)