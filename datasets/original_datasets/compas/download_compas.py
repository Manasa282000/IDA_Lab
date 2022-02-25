import pandas as pd
url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(url)
colnames = df.columns
df_new = df[colnames[5:18]]
df_new = pd.concat([df_new, df['c_charge_degree']], axis=1)
df_new = pd.concat([df_new, df['two_year_recid']], axis=1)
df_new = df_new.dropna()
df_new = df_new[['sex', 'age', 'juv_fel_count', 'juv_misd_count', 
                 'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid']]
 
def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            k = df[colnames[j]] > ts[j][s]
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X
df_new['sex=female'] = 1
k = df_new['sex']=='Male'
df_new.loc[k, 'sex=female'] = 0
df_new = df_new.drop('sex', axis=1)
 
df_new['current_charge_degree=felony'] = 1
k = df_new['c_charge_degree']!='F'
df_new.loc[k, 'current_charge_degree=felony'] = 0
df_new = df_new.drop('c_charge_degree', axis=1)
 
df_new['juvenile_crimes'] = df_new['juv_fel_count'] + df_new['juv_misd_count'] + df_new['juv_other_count']
df_new = df_new.drop('juv_other_count', axis=1)
 
colnames_new = ['sex=female', 'age', 'juv_fel_count', 'juv_misd_count',
       'juvenile_crimes', 'priors_count', 'current_charge_degree=felony',
       'two_year_recid']
df_new = df_new.reindex(columns = colnames_new)
df_new.to_csv('compas.csv', sep=";", index=False)
