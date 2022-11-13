import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
from scipy.stats import pearsonr
from scipy.stats import zscore

def plot_time_series(time_series, img_file):
	data_x = time_series['x']
	data_y = time_series['y']

	plt.plot(data_x, data_y, color='orange')
	plt.savefig(img_file)
	plt.close()

def plot_ts_tuple(ts_tuple, img_file):
	tuple_y = []
	tuple_x = []
	for i in range(len(ts_tuple)):
		tuple_x.append(ts_tuple[i][0])
		tuple_y.append(ts_tuple[i][1])
		
	plt.plot(tuple_x, tuple_y, color='blue')
	plt.savefig(img_file)
	plt.close()

# 简单的预处理，如果存在相同的两个x，则删除掉一个x。该部分对应在前端处理中，使用样条smooth处理掉
def process_sketch(sketch_data):
	del_index = []
	for i in range(0,len(sketch_data['x']) - 1):
		if sketch_data['x'][i] == sketch_data['x'][i+1]:
			del_index.append(i)
			
	cleaned_sketch = sketch_data.drop(index=del_index)
	cleaned_sketch.reset_index(drop=True, inplace=True)  # 重置索引来保证后续操作正确

	return cleaned_sketch

# 对草图数据进行normalization,将[600，400]的数据处理为(0,100)
def normalize_sketch(sketch_data):
	scale_x = 1/6 # [0, 600] -> [0, 100]
	scale_y = 1/4 # [0, 400] -> [0, 100]

	for i in range(len(sketch_data)):
		tmpx = sketch_data['x'][i]
		tmpx = tmpx * scale_x
		tmpy = sketch_data['y'][i]
		tmpy = tmpy * scale_y
		sketch_data['x'][i] = tmpx
		sketch_data['y'][i] = tmpy
	
	normalized_sketch = list(zip(sketch_data['x'], sketch_data['y']))# tmp格式为[(x1,y1),(x2,y2)]
	return normalized_sketch

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

def CalculatePearson(Y1, Y2):
    ZscoreY1 = zscore(Y1)
    ZscoreY2 = zscore(Y2)
    return pearsonr(ZscoreY1, ZscoreY2)[0]

def CalculateCosine(Y1, Y2):
    ZscoreY1 = zscore(Y1)
    ZscoreY2 = zscore(Y2)
    return np.dot(ZscoreY1, ZscoreY2) / (np.linalg.norm(ZscoreY1) * np.linalg.norm(ZscoreY2))

def ShapeError(Test_C,Test_Q,ComScale): # 输入应该为相同长度的sequence test_c = [(x1,y1),(x2,y2),(x3,y3)]
    c_y = []
    q_y = []
    for i in range(ComScale):
        c_y.append(Test_C[i][0]*Test_C[i][1])
        q_y.append(Test_Q[i][0]*Test_Q[i][1])
    height_c = max(c_y) - min(c_y)
    height_q = max(q_y) - min(q_y)
    global_y = height_c / height_q
    R_y = height_c / global_y * height_q
    ShapeError = 0
    for i in range(ComScale):
        ShapeError += abs((global_y * R_y * test_q[i][1] - test_c[i][1]) / height_c)
    ShapeError = ShapeError / ComScale
    return ShapeError

if __name__=="__main__":
	path = "/home/haojianing/sketch/raw_sketch_data" # 文件夹目录

	files = os.listdir(path) # 地道道文件夹下所有文件名称
	for file in files:
		if file.__contains__('csv'):
			position = os.path.join(path, file)
			data = pd.read_csv(file_path)
			img_file = file_path.replace('csv', 'png')
			plot_time_series(data, img_file)