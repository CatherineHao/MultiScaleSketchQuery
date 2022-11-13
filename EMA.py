# 不带偏差项，计算指数平滑平均
def calculateEMA(data,bate):
    ema_data = [0]
    for i in range(0, len(data)):
        vt = bate*ema_data[i] + (1-bate)*data[i]
        ema_data.append(vt)
    return ema_data