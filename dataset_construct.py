import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
from tqdm import tqdm

threshold = 17 # 当求导数的时候，使用16+1加速，使用原序列的时候使用16 = 2^4来完成加速

# 针对[0]为label，[1]为对应y值的ED计算代码
def EDdist(s1,s2):
    return math.sqrt(sum([(a[1]-b[1])**2 for (a,b) in zip(s1,s2)]))

def TakeLast(elem):
    return elem[1]

def EDDist(s1,s2): # 必须s1是针对时间序列的，s2是针对草图序列
    dist = 0
    for i in range(len(s1)):
        tmp1 = s1[i][1]
        tmp2 = s1[i][2]
        vec1 = np.array([tmp1,tmp2])
        vec2 = np.array(s2[i])
        dist = dist + np.sqrt(np.sum(np.square(vec1-vec2)))
    return dist

# 针对非段的'shape error'，为曼哈顿距离，输入为x[a,b],y[a,b]
def ManHattenDist(x,y): 
    return sum(map(lambda i,j:abs(i-j),x,y))

def DTWDistanceWithW(s1, s2, w):  # w表示窗口大小
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

# LTTB 代码
class LttbException(Exception):
    pass

def largest_triangle_three_buckets(data, threshold):
    """
    Return a downsampled version of data.
    Parameters
    ----------
    data: list of lists/tuples
        data must be formated this way: [[x,y], [x,y], [x,y], ...]
                                    or: [(x,y), (x,y), (x,y), ...]
    threshold: int
        threshold must be >= 2 and <= to the len of data
    Returns
    -------
    data, but downsampled using threshold
    """

    # Check if data and threshold are valid
    if not isinstance(data, list):
        raise LttbException("data is not a list")
    if not isinstance(threshold, int) or threshold <= 2 or threshold >= len(data):
        raise LttbException("threshold not well defined")
    for i in data:
        if not isinstance(i, (list, tuple)) or len(i) != 2:
            raise LttbException("datapoints are not lists or tuples")

    # Bucket size. Leave room for start and end data points
    every = (len(data) - 2) / (threshold - 2)

    a = 0  # Initially a is the first point in the triangle
    next_a = 0
    max_area_point = (0, 0)

    sampled = [data[0]]  # Always add the first point

    for i in range(0, threshold - 2):
        # Calculate point average for next bucket (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_rang_end = avg_range_end if avg_range_end < len(data) else len(data)

        avg_range_length = avg_rang_end - avg_range_start

        while avg_range_start < avg_rang_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        # Get the range for this bucket
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        # Point a
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1

        while range_offs < range_to:
            # Calculate triangle area over three buckets
            area = math.fabs(
                (point_ax - avg_x)
                * (data[range_offs][1] - point_ay)
                - (point_ax - data[range_offs][0])
                * (avg_y - point_ay)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1

        sampled.append(max_area_point)  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)

    sampled.append(data[len(data) - 1])  # Always add last

    return sampled

# 不带偏差项
def calculateEMA(data,bate):
    ema_pollutant = [0]
    for i in range(0, len(data)):
        vt = bate*ema_pollutant[i] + (1-bate)*data[i]
        ema_pollutant.append(vt)
    # print(ema_pm25)
    return ema_pollutant

# 最终输出
Final = []

# 读入需要匹配的时间序列数据
#LongTimeSeriesData = pd.read_csv('./test0919/QueryResult2016pm25.csv')
LongTimeSeriesData = pd.read_csv('./poll_data/60288.csv')
LongTimeSeriesDataY = LongTimeSeriesData['so2'].values.tolist()
LenX = len(LongTimeSeriesDataY)
TimeX = np.linspace(1,LenX,LenX)
TimeSeries = list(zip(TimeX,LongTimeSeriesDataY))

# 读取创建的20条测试数据集
path = "./points" # 文件夹目录
files = os.listdir(path) # 对应文件夹下所有文件名称
for file in tqdm(files):
    position = path + '\\' +file
    data = pd.read_csv(position)
    # 简单的预处理，如果存在相同的两个x，则删除掉一个x。该部分对应在前端处理中，使用样条smooth处理掉
    del_index = []
    for i in range(0,len(data['x'])-1):
        if data['x'][i] == data['x'][i+1]:
            del_index.append(i)
    data = data.drop(index=del_index)
    data.reset_index(drop=True,inplace=True)  # 重置索引来保证后续操作正确

    # 对数据进行normalization,将[600，400]的数据处理为(0,100)
    for i in range(len(data)):
        tmpx = data['x'][i]
        tmpx = tmpx / 6
        tmpy = data['y'][i]
        tmpy = tmpy / 4
        data['x'][i] = tmpx
        data['y'][i] = tmpy
    tmp = list(zip(data['x'],data['y']))# tmp格式为[(x1,y1),(x2,y2)]
    
    # 进行LTTB处理
    SegmentSequence = largest_triangle_three_buckets(tmp, threshold)

    # 进行(c,l)的转换
    print(SegmentSequence)
    DaoSequence = []
    for i in range (len(SegmentSequence)-1):
        temC = (SegmentSequence[i+1][1] - SegmentSequence[i][1])/(SegmentSequence[i+1][0] - SegmentSequence[i][0])
        temL = SegmentSequence[i+1][0] - SegmentSequence[i][0]
        DaoSequence.append(((temC,temL)))
        
    # 随机不同的 lttb threshold，来得到不同的对应时间序列数据进行匹配
    for i in range(50):
        '''需要EMA则对theta进行推算，并设置对应的threshold值
        theta = 0.3334 # 日0.3334 周0.904762 月 0.977778
        EmaTime = calculateEMA(LongTimeSeriesDataY,theta)
        Timehold = int(len(EmaTime) / 1.5) # 日对应1.5 周对应10.5 月对应45
        '''
        Timethreshold = random.randint(1500,len(TimeSeries))
        SegmentTime = largest_triangle_three_buckets(TimeSeries, Timethreshold)
        DaoTime = []
        for i in range (len(SegmentTime)-1):
            temC = (SegmentTime[i+1][1] - SegmentTime[i][1])/(SegmentTime[i+1][0] - SegmentTime[i][0])
            temL = SegmentTime[i+1][0] - SegmentTime[i][0]
            DaoTime.append(((int(SegmentTime[i][0]),temC,temL))) # (起点，tempc,tmpl)
            
        QueryScale = 16
        EDResults = []  # 记录查询ED结果
        SketchMappings = 3
        Blanks = 5
        # 计算欧式距离ED，得到与对应的最相似的三条子时间序列，观察是否相似
        for window in range(0, len(DaoTime) - QueryScale + 1):
            # 滑动窗口匹配数据
            s1 = DaoTime[window: window + QueryScale]
            s2 = DaoSequence # 需要匹配的草图数据
            # ED_dis = EDdist(s1,s2) # 原始只针对y的对比
            ED_dis = EDDist(s1,s2)
            EDResults.append([window,ED_dis])

        QueryResults = EDResults # 创建一个新的list进行排序
        # 按照相似性从高低排序
        QueryResults.sort(key=TakeLast)  # takeLast记录相似性  按照相似性进行排序
        # 返回最匹配的前num个结果
        TopResults = []
        # 第一个相似度最高因此必须添加
        TopResults.append(QueryResults[0])
        # 元素要和已经找到的依次比较  每次取出的一段元素
        for x in range(1, len(QueryResults)):
            # 假设当前元素满足题意
            flag = True
            for j in range(len(TopResults)):
                if abs(QueryResults[x][0] - TopResults[j][0]) <= Blanks:  # Blanks查找间距  类似步长
                    flag = False
                    break
            # 满足题意保留  匹配
            if flag == True:
                # 找到了指定数量的结果
                if len(TopResults) == SketchMappings:  # SketchMappings一个草图匹配三个
                    break
                TopResults.append(QueryResults[x])
        print("The Query results :", TopResults)

        YList= []# YList # 为对应的长度为16的 处理后的segementtime值
        for item in range(QueryScale):
            YList.append(SegmentTime[TopResults[0][0]:TopResults[0][0] + QueryScale][item][1])

        SketchTmp = ";".join([str(num) for num in DaoSequence])
        TimeTmp = ";".join([str(num) for num in YList])
        SimValue = []
        for item in range(3):
            YList = []
            for j in range(QueryScale):
                YList.append(SegmentTime[TopResults[item][0]:TopResults[item][0] + QueryScale][j][1])
            TimeTmp = ";".join([str(num) for num in YList])
            SketchTmp = ";".join([str(num) for num in DaoSequence])
            SimValue = 1-(math.exp(-TopResults[item][1]))
            Final.append([SketchTmp,TimeTmp,SimValue])

# 数据集构建完成，最后一步，导出为csv文件
name=['Sketch','MatchSeries','Sim']
TestData = pd.DataFrame(columns=name,data=Final)
#len(TestData)
TestData.to_csv('./TestDataSet.csv',index=False)