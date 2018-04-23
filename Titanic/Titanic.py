#coding=utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
pd.set_option('display.max_rows',None)
data_train = pd.read_csv("Train.csv")
#data_train.info()和data_train.info是两种不同的输出结果
#print(data_train.info())
#print(data_train.describe())

# PassengerId => 乘客ID
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱
# Embarked => 登船港口

data_train = pd.read_csv("train.csv")
from sklearn.ensemble import RandomForestRegressor


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=1500, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_format_data(df):
    # 看看家族是否够大，咳咳
    df['Family_Size'] = df['SibSp'] + df['Parch']

    df['AgeCat'] = df['Age']
    df.loc[(df.Age <= 10), 'AgeCat'] = 0
    df.loc[(df.Age > 60), 'AgeCat'] = 3
    df.loc[(df.Age > 10) & (df.Age <= 30), 'AgeCat'] = 3
    df.loc[(df.Age > 30) & (df.Age <= 60), 'AgeCat'] = 3

    df.loc[(df.Sex == 'female'), 'Sex'] = 0
    df.loc[(df.Sex == 'male'), 'Sex'] = 1

    df.loc[(df.Embarked == 'S'), 'Embarked'] = 0
    df.loc[(df.Embarked == 'Q'), 'Embarked'] = 1
    df.loc[(df.Embarked == 'C'), 'Embarked'] = 3
    df.loc[(df.Embarked.isnull()), 'Embarked'] = 0

    return df


#补全年龄
data_train, rfr = set_missing_ages(data_train)
#one hot 编码
#dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
#dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



data_train = set_format_data(data_train)
data_train = data_train.drop(['PassengerId', 'Name', 'Age', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare'], axis=1)
X = data_train.as_matrix()[:,1:]
y = list(data_train.as_matrix()[:,0])

# fit到RandomForestRegressor之中
from sklearn.ensemble import  RandomForestClassifier
rfr2 = RandomForestClassifier(random_state=0, n_estimators=1500, n_jobs=-1)
rfr2.fit(X, y)

data_test = pd.read_csv("test.csv")
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_format_data(data_test)
data_test = data_test.drop([ 'Name', 'Age', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare'], axis=1)
data_test['Survived'] = data_test['AgeCat']
predictions = data_test.as_matrix()[:,1:-1]
data_test['Survived'] = rfr2.predict(predictions)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':data_test['Survived'].as_matrix()})
result.to_csv("myself.csv", index=False)

print(data_test)

