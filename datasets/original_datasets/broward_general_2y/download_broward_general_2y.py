url = 'https://raw.githubusercontent.com/BeanHam/interpretable-machine-learning/master/broward/data/broward_data.csv'
import pandas as pd
df = pd.read_csv(url)
df_new = df.iloc[:,2:42]
df_new = df_new.drop('race', axis=1)
df_new.to_csv('broward_general_2y.csv', sep=";", index=False)
