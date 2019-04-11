#!/usr/bin/env python
# coding: utf-8

# goal of this code is to update sonar table with the latest data

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


def query_api(ticker, granularity, *dates):
     # function to query SONAR API, returns data frame with date and ticker values

    if len(dates) > 0:
        start_date = dates[0]
        end_date = dates[1]
        url_base = 'https://dev.freightwaves.com/public/TruckingIndexDataRaw.aspx?key=c419cfc6-9266-4003-974a-5e8e9cad8729&symbol' + \
            '={0}.{1}&date={2}&date_end={3}'
        url = url_base.format(ticker, granularity, start_date, end_date)
    else:
        url_base = 'https://dev.freightwaves.com/public/TruckingIndexDataRaw.aspx?key=c419cfc6-9266-4003-974a-5e8e9cad8729&symbol' + \
            '={0}.{1}'
        url = url_base.format(ticker, granularity)
    response = requests.get(url).text

    if len(response) < 100:
        return pd.DataFrame()

    else:
        data = json.loads(response)
        data = pd.DataFrame(data)

        data['date'] = pd.to_datetime(data['data_timestamp']).dt.date

        column_name = ticker+'_'+granularity
        data = data.rename(columns={'data_value': column_name})
        data[column_name] = data[column_name].astype(float)

        return data[['date', column_name]]


# In[164]:


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


# get existing SONAR data
sql_query = '''
select * from sonar_input
'''
sonar_input = get_sql(sql_query)
# in DB missing values are stores as 0
sonar_input = sonar_input.replace(0.0, np.nan)

end_date = str(datetime.now().year*10000 +
               datetime.now().month*100+datetime.now().day)

flat_file = pd.DataFrame()
for column in sonar_input.columns[2:]:

    existing_data = sonar_input[[column, 'date', 'id']].dropna()

    start_date = pd.to_datetime(existing_data['date'].max())
    start_date = start_date+timedelta(days=1)
    start_date = str(start_date.year*10000+start_date.month*100+start_date.day)

    last_id = existing_data['id'].max()
    start_id = last_id+1

    data = query_api(column.split('_')[0], column.split('_')[
                     1], start_date, end_date)
    data['id'] = start_id
    data['id'] = data['id']+data.index
    flat_file = pd.concat([flat_file, data], sort=True)

flat_file = pd.DataFrame(flat_file.groupby('date').max()).reset_index()
flat_file = flat_file.sort_values(by='date')
flat_file.to_csv(str(datetime.now().year*1000+datetime.now().month *
                     100+datetime.now().day) + r'.csv', encoding='utf-8', index=False)


for_upload = pd.concat(
    [flat_file, sonar_input[sonar_input['id'] >= flat_file['id'].min()].copy()], sort=True)
for_upload = pd.DataFrame(for_upload.groupby('date').max()).reset_index()
for_upload = for_upload.sort_values(by='id')
for_upload = for_upload.fillna(0.0)


update_new_rows_mysql(for_upload, 'sonar_input')
