from forest import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
import csv
import numpy as np
import os
import shutil

if __name__ == "__main__":
    data = pd.read_csv('sdss_redshift.csv')
    

    source_file = 'sdss.csv'
    copy_file = 'sdss_predict1.csv'
    shutil.copyfile(source_file, copy_file)
    
    
    
    x = data.drop(columns = 'redshift').to_numpy()
    y = data['redshift'].to_numpy()
    x_tr, x_t, y_tr, y_t = train_test_split(x, y, train_size=0.8,
                        random_state=42)
    regr = RandomForestRegressor(x_tr, y_tr, Ans = 1, rand = 0.7)
    
    regr.fit(x_tr, y_tr)
    y_p_t = regr.predict(x_t)
    y_p_tr = regr.predict(x_tr)

    
    plt.figure()
    plt.plot(np.repeat(y_t - y_p_t, 4),'*' , color='indigo', ms = 0.5, label='истинное значение — предсказание на Тесте')
    plt.plot(y_tr - y_p_tr,'o' ,ms = 0.4, color='orange', label='истинное значение — предсказание на обучающей')
    plt.legend() 
    plt.savefig('redhift.png') 
    plt.show()
    
    
    e1 = mean_squared_error(y_tr, y_p_tr)
    e2 = mean_squared_error(y_t, y_p_t)
    

    ans = {"train": e1,"test": e2}
    with open('redhsift.json', 'w') as f:
        json.dump(ans, f)

    df = pd.read_csv('sdss.csv')
    x = df.to_numpy()
    y_pred =regr.predict(x)

    dk = pd.read_csv('sdss_predict1.csv')
    dk['redshift'] = y_pred
    
    dk.to_csv('sdss_predict1.csv')   
 
