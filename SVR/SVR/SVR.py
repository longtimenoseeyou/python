import pandas as pd
import matplotlib.pylab as plt
from sklearn.svm import SVR
import sklearn.preprocessing as preprocessing

df = pd.read_excel('C:/Users/cwt/Desktop/学习/数据挖掘/作业3数据拟合回归作业/附件2.PA.xls')
df = pd.DataFrame(df)  #导入数据
df = pd.DataFrame(df.values.T) #数据转置
y=[]
for column in range(28):
    for row in range(1,97):
        y.append(df[column][row]) #将数据导出一列
ds= pd.date_range(start='2006-05-10 00:00:00',end='2006-06-06 23:45:00',freq='15T') #按15min间隔生成时间序列
df1= pd.DataFrame(y,index=ds)

#标准化
for i in range(len(df1)):
    df1.values[i] = (df1.values[i] - df1.values[:].min() / (df1.values[:].max() - df1.values[:].min()))

#7天(672)预测
x_train=ds[:-672].values
y_train=df1[:-672].values
x_test=ds[2016:2112].values
y_test=df1[2016:2112].values


lin_svr=SVR(kernel='linear')
lin_svr.fit(x_train.reshape(-1,1),y_train.reshape(-1,1).ravel())
lin_svr_pred=llin_svr.predict(x_test)

plt.figure(figsize=(16,8),dpi = 100)
plt.grid(linestyle='--')
plt.plot( y_test, label='actual value')
plt.plot( lin_svr_pred, label='predict value')
plt.legend(loc=1)
plt.show()
