from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts

# 从tushare中获取今年来黄金ETF的价格，并按照线性回归算法需求整理其中的数据
goldPrice = ts.get_k_data('518880', '2017-04-01', '2019-10-14')  # tushare只提供了2017年4月之后的数据
goldPrice = goldPrice[['date', 'close']]  # 只截取其中的date和close数据列
# plt.figure()
# plt.plot(goldPrice.date, goldPrice.close)
# plt.ylabel("Gold ETF Prices")
# plt.show()
goldPrice['s_3'] = goldPrice['close'].shift(1).rolling(window=3).mean()  # 新增一列前3天的收盘平均值并赋值
goldPrice['s_7'] = goldPrice['close'].shift(1).rolling(window=7).mean()  # rolling+shift(1)指不包括当天往前7天的平均值
goldPrice = goldPrice.dropna()  # tushare是没有空值的，但前3天和前7天的平均值会产生空值，所以要去除
goldPrice = goldPrice.set_index('date')  # 设置goldPrice的默认索引为date列

# 从goldPrice中提取训练数据和测试数据
x = goldPrice[['s_3', 's_7']]
y = goldPrice[['close']]
t = .8
t = int(len(goldPrice)*t)
x_train = x[:t]
y_train = y[:t]
x_test = x[t:]
y_test = y[t:]

# 训练/测试并并预测最新一天的收盘价
linear = LinearRegression().fit(x_train, y_train)  # 建立训练模型，获取训练数据
predicted_price = linear.predict(x_test)  # 从给定的测试数据中得到测试结果
predicted_price = pd.DataFrame(predicted_price, index=y_test.index+1, columns=['predicted_price'])
ax = predicted_price[-5:].plot(color='g')
y_test[-5:].plot(color='r', ax=ax)
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel('Gold ETF Price')
plt.show()
print(predicted_price.tail())

print('Welcome to Git world!!!')