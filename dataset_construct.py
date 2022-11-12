import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
from tqdm import tqdm

import utils
import LTTB

# def TakeLast(elem):
#     return elem[1]

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

# # 针对非段的'shape error'，为曼哈顿距离，输入为x[a,b],y[a,b]
# def ManHattenDist(x,y): 
#     return sum(map(lambda i,j:abs(i-j),x,y))

# def DTWDistanceWithW(s1, s2, w):  # w表示窗口大小
#     DTW = {}

#     w = max(w, abs(len(s1) - len(s2)))

#     for i in range(-1, len(s1)):
#         for j in range(-1, len(s2)):
#             DTW[(i, j)] = float('inf')
#     DTW[(-1, -1)] = 0

#     for i in range(len(s1)):
#         for j in range(max(0, i - w), min(len(s2), i + w)):
#             dist = (s1[i] - s2[j]) ** 2
#             DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

#     return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

# # LTTB 代码
# class LttbException(Exception):
#     pass



# # 不带偏差项
# def calculateEMA(data,bate):
#     ema_pollutant = [0]
#     for i in range(0, len(data)):
#         vt = bate*ema_pollutant[i] + (1-bate)*data[i]
#         ema_pollutant.append(vt)
#     # print(ema_pm25)
#     return ema_pollutant

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

lttb_thre = 17

if __name__=="__main__":
    raw_data = read_raw_data(raw_data_files)

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
            for i in range(50):
                '''需要EMA则对theta进行推算，并设置对应的threshold值
                theta = 0.3334 # 日0.3334 周0.904762 月 0.977778
                EmaTime = calculateEMA(LongTimeSeriesDataY,theta)
                Timehold = int(len(EmaTime) / 1.5) # 日对应1.5 周对应10.5 月对应45
                '''
                time_thre = random.randint(1500, len(raw_data))
                lttb_raw_data = LTTB.largest_triangle_three_buckets(raw_data, time_thre)
                derivative_raw_data = convert_to_derivative(lttb_raw_data)
            
                QueryScale = len(derivative_sketch)
                EDResults = []  # 记录查询ED结果
                SketchMappings = 3
                Blanks = 5

                # 计算欧式距离ED，得到与对应的最相似的三条子时间序列，观察是否相似
                for window in range(0, len(derivative_raw_data) - QueryScale + 1):
                    # 滑动窗口匹配数据
                    s1 = derivative_raw_data[window: window + QueryScale]

                    # ED_dis = EDdist(s1,s2) # 原始只针对y的对比
                    ED_dist = EDDist(s1, derivative_sketch)
                    EDResults.append([window, ED_dist])

                # QueryResults = EDResults # 创建一个新的list进行排序
                # # 按照相似性从高低排序
                # QueryResults.sort(key=TakeLast)  # takeLast记录相似性  按照相似性进行排序
                # # 返回最匹配的前num个结果
                # TopResults = []
                # # 第一个相似度最高因此必须添加
                # TopResults.append(QueryResults[0])
                # # 元素要和已经找到的依次比较  每次取出的一段元素
                # for x in range(1, len(QueryResults)):
                #     # 假设当前元素满足题意
                #     flag = True
                #     for j in range(len(TopResults)):
                #         if abs(QueryResults[x][0] - TopResults[j][0]) <= Blanks:  # Blanks查找间距  类似步长
                #             flag = False
                #             break
                #     # 满足题意保留  匹配
                #     if flag == True:
                #         # 找到了指定数量的结果
                #         if len(TopResults) == SketchMappings:  # SketchMappings一个草图匹配三个
                #             break
                #         TopResults.append(QueryResults[x])
                # print("The Query results :", TopResults)
                
                # # 计算每一个的simvalue，QueryResults
                # MaxED = QueryResults[-1][1]
                # MinED = QueryResults[0][1]
                # for x in range(0,len(QueryResults)):
                #     # 归一化后，sim = 1-dis,sim=[0,1]
                #     QueryResults[x][1] = (QueryResults[x][1]-MinED) / (MaxED-MinED)
                #     QueryResults[x][1] = 1-QueryResults[x][1]
                # #print('QueryResults:',QueryResults)

                # #第一列，存储查询序列的c,l,第二列，查询到的数据SegementTime对应的y值,第三列存储similarity
                # SketchTmp = ";".join([str(num) for num in DaoSequence])
                # for item in range(0,len(QueryResults)):
                #     YList = [] # YList为对应的长度为16的处理后的segementtime值
                #     for j in range(QueryScale):
                #         YList.append(SegmentTime[QueryResults[item][0]:QueryResults[item][0] + QueryScale][j][1])
                #     TimeTmp = ";".join([str(num) for num in YList])
                #     print(TimeTmp)
                #     print(SketchTmp)            
                #     SimValue = QueryResults[item][1]
                #     print(SimValue)
                #     Final.append([SketchTmp,TimeTmp,SimValue])

# # 数据集构建完成，最后一步，导出为csv文件
# name=['Sketch','MatchSeries','Sim']
# TestData = pd.DataFrame(columns=name,data=Final)
# #len(TestData)
# TestData.to_csv('./TestDataSet.csv',index=False)