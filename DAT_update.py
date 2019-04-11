#!/usr/bin/env python
# coding: utf-8

# In[1]:


# goal of this code is to update sonar and dat tables with the latest data


import json
import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os


def get_sql(sql_query):
    # function to query internal DB
    import mysql.connector as sql

    db_connection = sql.connect(
        host='157.230.166.29', database='lanecaster_db', user='root', password='lanecaster1!')
    df = pd.read_sql(sql_query, con=db_connection)
    db_connection.close()

    return df


# In[165]:


def update_new_rows_mysql(pandas_dataframe, table_name):
     # function to update the dataframe to internal DB
    import mysql.connector as sql

    # change the connection parameters if needed
    db_connection = sql.connect(
        host='157.230.166.29', database='lanecaster_db', user='root', password='lanecaster1!')
    mycursor = db_connection.cursor(buffered=True)

    # formating date column as string
    if 'date' in pandas_dataframe.columns:
        pandas_dataframe['date'] = pandas_dataframe['date'].astype(str)
        pandas_dataframe['date'] = pandas_dataframe['date'].apply(
            lambda x: "'"+x+"'")

    for index, row in pandas_dataframe.iterrows():
        column_names = [x for x in pandas_dataframe.columns]
        values = pandas_dataframe.loc[index].values

        update_string = ''
        for column_index, column_name in enumerate(column_names):
            if column_name != 'id':
                update_string = update_string+'\n'
                update_string = update_string+column_name + \
                    '='+str(values[column_index])+','

        column_names = ','.join(map(str, column_names))
        values = ','.join(map(str, values))

        sql_query = '''
        INSERT INTO 
         {0}({1})
        VALUES
         ({2})

         ON DUPLICATE KEY UPDATE

         {3};
        '''.format(table_name, column_names, values, update_string)
        sql_query = sql_query.replace(',;', ' ;')
        mycursor.execute(sql_query)
        db_connection.commit()

    # closing the connection
    db_connection.close()


# CHONG CAN YOU PLEASE CHANGE THIS PATH
folder = r'/var/www/DAT_data'
# folder=r'C:\Users\vukan.vujic\Documents\work\work\sonar\january\packing_dec_22\helper_files\DAT'


# In[176]:


DAT_SONAR_mapping = {'CHICAGO': 'CHI', 'HOUSTON': 'HOU', 'PHILADELPHIA': 'PHL', 'MINNEAPOLIS': 'MSP',
                     'DALLAS': 'DAL', 'MEMPHIS': 'MEM', 'COLUMBUS': 'CMH', 'LOUISVILLE': 'SDF', 'CHARLOTTE': 'CLT',
                     'DETROIT': 'DTW', 'ST LOUIS': 'STL', 'ATLANTA': 'ATL', 'BALTIMORE': 'BWI', 'MILWAUKEE': 'MKE',
                     'JACKSONVILLE': 'JAX', 'NEW ORLEANS': 'MSY', 'LOS ANGELES': 'LAX', 'SEATTLE': 'SEA'}


# get existing DAT data
sql_query = '''
select * from dat_input
'''
dat_input = get_sql(sql_query)


# In[182]:


len(dat_input.columns)


# In[ ]:


start_date = pd.to_datetime(dat_input['date'].max())
start_date = start_date.year*10000+start_date.month*100+start_date.day


# In[ ]:


start_id = dat_input['id'].max()+1


# In[181]:


DAT = pd.DataFrame()
for file in [x for x in os.listdir(folder) if int(x.split('.')[0]) > start_date]:
    DAT_part = pd.read_csv(folder+r'/'+file, encoding='utf-8')
    DAT_part = DAT_part.replace(
        {'Orig City': DAT_SONAR_mapping, 'Dest City': DAT_SONAR_mapping})
    DAT_part['lane'] = DAT_part[['Orig City', 'Dest City']].apply(
        lambda x: x[0]+x[1], axis=1)
    DAT_part = DAT_part[['lane', 'PC-Miler Practical Mileage', 'Spot Avg Linehaul Rate',
                         'Spot Low Linehaul Rate',    'Spot High Linehaul Rate', 'Spot Fuel Surcharge']].copy()
    DAT_part['date'] = pd.to_datetime(file.split('.')[0]).date()
    DAT = pd.concat([DAT, DAT_part])


# In[184]:


DAT = pd.pivot_table(DAT, index='date', columns='lane',
                     values='Spot Avg Linehaul Rate').reset_index()
DAT = DAT.fillna(0.0)
DAT['id'] = start_id+DAT.index


update_new_rows_mysql(DAT, 'dat_input')
