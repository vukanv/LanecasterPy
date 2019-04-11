#!/usr/bin/env python
# coding: utf-8

# goal of this code is to update sonar table with the latest data

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import mysql.connector as sql


def get_access_token():
    url = "https://api.freightwaves.com/Credential/authenticate"
    data = {
        "username": "MergeTransitUser",
        "password": "F74g6eqPM@?g&tKs"
    }
    res_data = requests.post(url, json=data).json()['token']
    return res_data


def query_api(token, ticker, granularity, *dates):
     # function to query SONAR API, returns data frame with date and ticker values
    start_date = dates[0]
    end_date = dates[1]
    url_base = 'https://api.freightwaves.com/data/{0}/{1}/{2}/{3}'
    url = url_base.format(ticker, granularity, start_date, end_date)
    r = requests.get(url,  headers={"Authorization": "Bearer " + token})

    if r.status_code == 200:
        response = r.text
        data = json.loads(response)
        data = pd.DataFrame(data)

        data['date'] = pd.to_datetime(data['data_Timestamp']).dt.date

        column_name = ticker + '_' + granularity
        data = data.rename(columns={'data_Value': column_name})
        data[column_name] = data[column_name].astype(float)
        return data[['date', column_name]]

    else:
        return pd.DataFrame()


# In[164]:


def get_sql(sql_query):
    # function to query internal DB

    db_connection = sql.connect(
        host='157.230.166.29', database='lanecaster_db', user='root', password='lanecaster1!')
    # db_connection = sql.connect(
    #     host='localhost', database='lanecaster_db', user='root', password='')
    df = pd.read_sql(sql_query, con=db_connection)
    db_connection.close()

    return df


# In[165]:


def update_new_rows_mysql(pandas_dataframe, table_name):
     # function to update the dataframe to internal DB

    # change the connection parameters if needed
    db_connection = sql.connect(
        host='157.230.166.29', database='lanecaster_db', user='root', password='lanecaster1!')
    # db_connection = sql.connect(
    #     host='localhost', database='lanecaster_db', user='root', password='')
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


# get existing SONAR data
sql_query = '''
select * from sonar_input
'''
sonar_input = get_sql(sql_query)
# in DB missing values are stores as 0
sonar_input = sonar_input.replace(0.0, np.nan)
# end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
end_date = (datetime.today()).strftime('%Y-%m-%d')

flat_file = pd.DataFrame()
token = get_access_token()
for column in sonar_input.columns[2:]:

    existing_data = sonar_input[[column, 'date', 'id']].dropna()

    last_date = pd.to_datetime(existing_data['date'].max())
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_id = existing_data['id'].max() + 1

    data = query_api(token, column.split(
        '_')[0], column.split('_')[1], start_date, end_date)
    data['id'] = start_id
    data['id'] = data['id'] + data.index
    flat_file = pd.concat([flat_file, data], sort=True)
    print("get data " + start_date + "~" + end_date,
          column.split('_')[0], column.split('_')[1])

flat_file = pd.DataFrame(flat_file.groupby('date').max()).reset_index()
flat_file = flat_file.sort_values(by='date')
before_date = datetime.now() - timedelta(days=1)
flat_file.to_csv(str(before_date.year*10000 + before_date.month *
                     100+before_date.day) + r'.csv', encoding='utf-8', index=False)


for_upload = pd.concat(
    [flat_file, sonar_input[sonar_input['id'] >= flat_file['id'].min()].copy()], sort=True)
for_upload = pd.DataFrame(for_upload.groupby('date').max()).reset_index()
for_upload = for_upload.sort_values(by='id')
for_upload = for_upload.fillna(0.0)


update_new_rows_mysql(for_upload, 'sonar_input')
