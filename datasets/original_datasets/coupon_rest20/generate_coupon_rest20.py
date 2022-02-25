import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('../coupon/coupon_source.csv')#'C:/Users/chudi/.spyder2-py3/gosdt/data/coupon/coupon_source.csv')
data = data.drop('car', axis=1)
data = data.dropna()
data = data.drop('map', axis=1)
data = data.drop('click', axis=1)
data = data.drop('toCoupon_GEQ5min', axis=1)
data = data.drop('direction_opp', axis=1)
 
def data_transform(data):
    for i in range(data.shape[1]-1):
        le = preprocessing.LabelEncoder()
        le.fit(data.iloc[:,i])
        print(le.classes_)
        data.iloc[:,i] = le.transform(data.iloc[:,i])
    
    
    colnames = data.columns
    for i in range(len(colnames)-1):
        uni = data[colnames[i]].unique()
        uni.sort()
        for j in range(len(uni)-1):
            data[colnames[i]+str(uni[j])] = 1
            k = data[colnames[i]] != uni[j]
            data.loc[k, colnames[i]+str(uni[j])] = 0 
        data = data.drop(colnames[i], axis=1)
    data['target'] = data['Y']
    data = data.drop('Y', axis=1)
    return data
 
rest20 = data[data['coupon'] == 'Restaurant(<20)']
rest20 = rest20.drop('coupon', axis=1)
rest20 = data_transform(rest20)
rest20.to_csv('coupon_rest20.csv', sep=";", index=False)

