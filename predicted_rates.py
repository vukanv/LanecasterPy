#!/usr/bin/env python
# coding: utf-8


# this code, extracts inputs and fits a model on a preferred route (DALLAX)
# then extracts data for other routes and fits a model for them
# makes a 7day future prediction based on model and forecast and the past DAT data

# importing libraries
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
import os

import statsmodels.formula.api as smf
from itertools import permutations


def get_sql(sql_query):
    import mysql.connector as sql
    # function to query internal DB
    db_connection = sql.connect(
        host='157.230.166.29', database='lanecaster_db', user='root', password='lanecaster1!')
    df = pd.read_sql(sql_query, con=db_connection)
    db_connection.close()
    return df


# In[4]:


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


def gen_route_data(route, db_table, train=1):
    # sonar indicators to use as an explanatory variables
    variables = ['OTRI', 'VTRI', 'TRUK', 'HAUL',
                 'OTVI', 'ITVI', 'ITRI', 'OTMS', 'ITMS', 'TLT']
    route_variables = [x+'_'+route[0:3] for x in variables]
    route_variables.extend([x+'_'+route[-3:] for x in variables])

    if train == 1:
        route_variables.append('DATVF_'+route)

    route_variables.append('date')
    route_variables.extend(['GOEX_USA', 'GOIM_USA', 'GASPRC_USA'])

    sql_query = '''
    select {0} from sonar_input
    '''
    sql_query = sql_query.format(
        ', '.join('{0}'.format(w) for w in route_variables))

    data = get_sql(sql_query)
    data = data.replace(0.00, np.nan)

    # generating features
    # dates
    data['date'] = pd.to_datetime(data['date'])

    data['year'] = data['date'].dt.year
    data['week'] = data['date'].dt.week
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month

    # month end, firms ship loads in order to increase monthly revenues
    data['month_end'] = data['date']+MonthEnd(1)
    data['month_end'] = pd.to_datetime(data['month_end'])
    data['days_to_month_end'] = (data['month_end']-data['date']).dt.days
    data['month_end'] = data['days_to_month_end'].apply(
        lambda x: 1 if x <= 5 else 0)

    data['date'] = data['date'].dt.date
    data = data.sort_values(by='date')
    data = data.set_index('date')

    for variable in ['GOEX_USA', 'GOIM_USA', 'GASPRC_USA']:
        data[variable] = data[variable].fillna(method='ffill')

     # fixing dips in TRUK and HAUL?

    for variable in [x for x in data.columns if ('TRUK' in x) | ('HAUL' in x)]:
        # keeping original to know when the holidays were
        data[variable+'_slump'] = 0
        data['7dma'] = data[variable].rolling(14).mean()
        data['7dma'] = data['7dma'].fillna(method='bfill')
        data['change'] = data[variable]/data['7dma']
        data.loc[data['change'] < 0.8, variable] = np.nan
        data.loc[data['change'] < 0.8, variable+'_slump'] = 1
        data[variable] = data[variable].interpolate()

    data = data.drop(['7dma', 'change'], 1)

    functions = [np.mean, np.min, np.max, np.median, np.std]
    function_names = ['mean', 'min', 'max', 'median', 'stdev']

    # interpolating
    if train == 1:
        data['DATVF_'+route] = data['DATVF_'+route].interpolate()

    # using origin and destination instead of city names so that it can be extended
    data.columns = [x.replace(route[0:3], 'ORIGIN')
                    if route not in x else x for x in data.columns]
    data.columns = [x.replace(route[-3:], 'DESTINATION')
                    if route not in x else x for x in data.columns]

    # marking two days before and after holiday as holiday period (tend to have increased rates)
    data = data.reset_index()
    for date_index in data[data['TRUK_ORIGIN_slump'] == 1].index:
        for index in range(-2, 3):
            if index in data.index:
                data.loc[date_index+index, 'TRUK_ORIGIN_slump'] = 1
    data = data.set_index('date')

    explanatory_variables = [x for x in data.columns if (
        ('ORIGIN' in x) | ('DESTINATION' in x)) & ('slump' not in x)]

    if train == 1:
        explanatory_variables.append('DATVF_'+route)

    for variable in explanatory_variables:
        data[variable+'_change'] = data[variable]-data[variable].shift(1)
        data[variable+'_change_percent'] = (
            data[variable+'_change']/data[variable].shift(1)-1)*100
        data[variable+'_change7'] = data[variable]-data[variable].shift(7)
        data[variable +
             '_change7_percent'] = (data[variable]/data[variable].shift(7)-1)*100
        for index, function in enumerate(functions):
            for period in [7, 14]:
                data[variable+'_'+str(period)+'_'+function_names[index]
                     ] = data[variable].rolling(period).agg(function)
                data[variable+'_'+str(period)+'_'+function_names[index]+'_change'] = data[variable+'_'+str(
                    period)+'_'+function_names[index]] - data[variable+'_'+str(period)+'_'+function_names[index]].shift(period)

                data[variable+'_'+str(period)+'_'+function_names[index]+'_change_shift1'] = data[variable+'_'+str(
                    period)+'_'+function_names[index]] - data[variable+'_'+str(period)+'_'+function_names[index]].shift(1)

                data[variable+'_'+str(period)+'_'+function_names[index]+'_change_percent'] = (data[variable+'_'+str(
                    period)+'_'+function_names[index]]/data[variable+'_'+str(period)+'_'+function_names[index]].shift(period)-1)*100

                data[variable+'_'+str(period)+'_'+function_names[index]+'_change_percent_shift1'] = (data[variable+'_'+str(
                    period)+'_'+function_names[index]]/data[variable+'_'+str(period)+'_'+function_names[index]].shift(1)-1)*100

                data[variable+'_'+str(period)+'_'+function_names[index]+'_up'] = data[variable+'_'+str(
                    period)+'_'+function_names[index]+'_change'].apply(lambda x: 1 if x > 0 else 0)

    for variable in explanatory_variables:
        for index, function in enumerate(functions):
            data[variable+'_'+function_names[index]+'_cross'] = data[[variable+'_7_'+function_names[index],
                                                                      variable+'_14_'+function_names[index]]].apply(lambda x: 1 if x[0] > x[1] else 0, axis=1)

    # interactions
    interact_variables_1 = '_change7'
    interact_variables_2 = '_change'

    explanatory_variables = ['OTRI_ORIGIN', 'OTRI_DESTINATION', 'VTRI_ORIGIN', 'VTRI_DESTINATION',    'TRUK_ORIGIN', 'TRUK_DESTINATION', 'HAUL_ORIGIN', 'HAUL_DESTINATION', 'OTVI_ORIGIN', 'OTVI_DESTINATION',
                             'OTMS_ORIGIN', 'ITMS_ORIGIN', 'OTMS_DESTINATION', 'ITMS_DESTINATION',        'ITVI_DESTINATION', 'ITRI_DESTINATION', 'TLT_DESTINATION',          'ITVI_ORIGIN', 'ITRI_ORIGIN', 'TLT_ORIGIN']
    # explanatory_variables_base=explanatory_variables.copy()
    explanatory_variables_base = list(explanatory_variables)
    explanatory_variables = (
        [x+interact_variables_1 for x in explanatory_variables_base])
    explanatory_variables.extend(
        [x+interact_variables_2 for x in explanatory_variables_base])
    explanatory_variables.extend(['month_end', 'dayofweek'])

    # interactions?

    combinations_exp = permutations(explanatory_variables, 2)

    for combination in combinations_exp:
        # print(combination[0]+'_'+combination[1])
        data[combination[0]+'_'+combination[1]
             ] = data[combination[0]]*data[combination[1]]
        explanatory_variables.append(combination[0]+'_'+combination[1])

    return data


def gen_prediction_data(route, days_prediction=7, holidays=[pd.to_datetime('2019-01-11').date()]):
    # route, route for which rate is predicted
    # days_prediction number of days in the future for which prediction goes
    # holidays list of holiday dates, 2  days before and after holiday are marked as TRUCK_slump and act as an explanatory variable

    data_predict = gen_data(route, db_table, 0)
    variables = []
    for variable, lag in model_specification.items():
        data_predict[variable] = data_predict[variable].shift(lag-5)
        variables.append(variable)

    data_predict = data_predict.fillna(method='ffill')
    data_predict = data_predict.iloc[-days_prediction:][variables].copy()

    last_date = data_predict.index[-1]
    data_predict = data_predict.reset_index(drop=True)

    indices = []
    for index in range(0, days_prediction):
        new_date = last_date+timedelta(days=index)
        indices.append(new_date)
        month_end = pd.to_datetime(new_date+MonthEnd(1))
        days_to_month_end = (month_end-pd.to_datetime(new_date)).days
        if days_to_month_end < 5:
            data_predict.loc[index, 'month_end'] = 1

    data_predict.index = indices

    # remove later on
    if 'TRUK_ORIGIN_slump' not in data_predict.columns:
        data_predict['TRUK_ORIGIN_slump'] = np.nan

    for holiday in holidays:
        if holiday in data_predict.index:
            data_predict.loc[holiday, 'TRUK_ORIGIN_slump'] = 1

    data_predict = data_predict.reset_index()
    for date_index in data_predict.iloc[-6:][data_predict['TRUK_ORIGIN_slump'] == 1].index:
        for index in range(-2, 3):
            if (date_index+index) in data_predict.index:
                data_predict.loc[date_index+index, 'TRUK_ORIGIN_slump'] = 1
    data_predict = data_predict.set_index('index')
    return data_predict


# # FITTING THE MODEL


# parameters

# route to fit model on (try using the average instead??)
model_route = 'DALLAX'


# model specifications and the dependent variable, so that it knows what to fit on

dependent_variable = 'DATVF_'+model_route+'_change7'

model_specification_1 = {'HAUL_ORIGIN_14_mean_change_shift1': 1,  'ITRI_DESTINATION_14_mean_change_percent_shift1': 1,  'ITVI_DESTINATION_14_min_sq': 1,  'OTMS_ORIGIN_14_min_change_sq': 1,  'OTRI_DESTINATION_7_mean_change_percent_shift1': 1,
                         'OTVI_ORIGIN_14_min_sq': 1,  'TLT_ORIGIN_change_HAUL_ORIGIN_change7_sq': 1,  'TRUK_DESTINATION_14_min_sq': 1,  'VTRI_ORIGIN_change7': 1,  'VTRI_ORIGIN_change_ITMS_ORIGIN_change_sq': 1, 'month_end': 1, 'TRUK_ORIGIN_slump': 0}

model_specification_2 = {
    'HAUL_DESTINATION_7_median_change_percent_shift1': 2,
    'ITMS_DESTINATION_14_stdev_change_sq': 2,
    'ITMS_ORIGIN_change7_OTRI_DESTINATION_change_sq': 2,
    'ITRI_DESTINATION_14_max_change_percent_sq': 2,
    'ITVI_DESTINATION_14_min_sq': 2,
    'OTMS_ORIGIN_14_min_change_shift1': 2,
    'OTVI_DESTINATION_change7_TRUK_ORIGIN_change7': 2,
    'TLT_DESTINATION_change_TRUK_ORIGIN_change7_sq': 2,
    'TRUK_ORIGIN_change_VTRI_DESTINATION_change_sq': 2,
    'TRUK_ORIGIN_change_sq': 2, 'month_end': 2, 'TRUK_ORIGIN_slump': 0}

model_specification_3 = {'HAUL_ORIGIN_7_median_change': 3,  'ITMS_ORIGIN_7_stdev_change': 3,  'ITMS_ORIGIN_change7_TLT_ORIGIN_change7_sq': 3,  'ITRI_DESTINATION_14_mean_change_sq': 3,  'ITVI_DESTINATION_14_mean_sq': 3,
                         'OTVI_ORIGIN_7_min': 3,  'TRUK_DESTINATION_change_OTRI_DESTINATION_change7_sq': 3,  'TRUK_ORIGIN_change_OTMS_DESTINATION_change': 3,  'VTRI_DESTINATION_change7_TRUK_DESTINATION_change_sq': 3, 'month_end': 3, 'TRUK_ORIGIN_slump': 0}

model_specification_4 = {'HAUL_DESTINATION_14_min_change_percent': 4,  'ITMS_DESTINATION_7_stdev_change_sq': 4,  'OTMS_DESTINATION_change7_ITRI_ORIGIN_change7': 4,  'OTMS_DESTINATION_change7_TLT_DESTINATION_change7': 4,  'OTMS_DESTINATION_change_VTRI_DESTINATION_change7': 4,
                         'OTRI_DESTINATION_change7_OTMS_DESTINATION_change': 4,  'OTVI_ORIGIN_change_TRUK_DESTINATION_change': 4,  'TRUK_ORIGIN_change_ITVI_ORIGIN_change': 4,  'VTRI_DESTINATION_change7_ITMS_DESTINATION_change7': 4, 'month_end': 4, 'TRUK_ORIGIN_slump': 0}

model_specification_5 = {'HAUL_DESTINATION_7_min_change_percent': 5,  'ITMS_DESTINATION_14_stdev': 5,  'ITMS_DESTINATION_change7_ITRI_DESTINATION_change_sq': 5,  'ITMS_DESTINATION_change7_VTRI_ORIGIN_change7_sq': 5,  'ITVI_ORIGIN_7_mean_change_percent': 5,
                         'OTMS_DESTINATION_change7_TLT_DESTINATION_change': 5,  'OTMS_ORIGIN_14_stdev_change_sq': 5,  'OTRI_ORIGIN_change_OTMS_ORIGIN_change7': 5,  'OTVI_ORIGIN_7_mean_change_percent': 5,  'TRUK_ORIGIN_14_median_change_shift1': 5, 'month_end': 5, 'TRUK_ORIGIN_slump': 0}

model_specification_6 = {'HAUL_ORIGIN_7_stdev_change_percent_sq': 6,  'ITMS_DESTINATION_14_mean_sq': 6,  'ITMS_DESTINATION_change7_OTRI_ORIGIN_change7': 6,  'ITMS_ORIGIN_change_VTRI_ORIGIN_change': 6,  'ITRI_ORIGIN_change7_OTMS_DESTINATION_change7_sq': 6,
                         'OTMS_DESTINATION_14_median_sq': 6,  'OTVI_ORIGIN_change7_percent': 6,  'TLT_DESTINATION_14_stdev_sq': 6,  'TRUK_DESTINATION_7_mean_change_sq': 6,  'TRUK_DESTINATION_change7_ITVI_DESTINATION_change_sq': 6, 'month_end': 6, 'TRUK_ORIGIN_slump': 0}

model_specification_7 = {'HAUL_ORIGIN_14_stdev_change_percent': 7,  'ITMS_DESTINATION_sq': 7,  'ITVI_DESTINATION_change_VTRI_DESTINATION_change': 7,  'OTMS_DESTINATION_14_max_sq': 7,  'OTMS_ORIGIN_change7_ITRI_ORIGIN_change7': 7,
                         'OTRI_ORIGIN_change7_TRUK_DESTINATION_change7': 7,  'OTVI_ORIGIN_change_percent_sq': 7,  'TLT_DESTINATION_7_max': 7,  'TRUK_DESTINATION_14_stdev_change': 7, 'month_end': 7, 'TRUK_ORIGIN_slump': 0}

models_dict = {'model_specification_1': model_specification_1, 'model_specification_2': model_specification_2,            'model_specification_3': model_specification_3, 'model_specification_4':
               model_specification_4,            'model_specification_5': model_specification_5, 'model_specification_6': model_specification_6,            'model_specification_7': model_specification_7}


# In[8]:


# generating data to fit the model for the representative route on
data_fit = gen_route_data(model_route, 'sonar_input')

# is this needed, sklearn cannot handle inf, but why do they appear in the first place, proabably when stdev taken?
data_fit = data_fit.replace([np.inf, -np.inf], 0)

# TRAINING EACH OF THE SEVEN MODELS
# 1 FOR EACH DAY IN THE FUTURE
trained_models = {}
for model_index in range(1, 8):

    model_specification = models_dict['model_specification_'+str(model_index)]
    data_train_model = data_fit.copy()

    variables = []
    for variable, lag in model_specification.items():
        if '_sq' in variable:
            data_train_model[variable] = data_train_model[variable.replace(
                '_sq', '')]**2
        data_train_model[variable] = data_train_model[variable].shift(lag)
        variables.append(variable)
    variables.append(dependent_variable)
    variables.remove(dependent_variable)

    trained_models['model_specification_'+str(model_index)] = smf.ols('{0}'.format(
        dependent_variable) + '~{0}'.format('+ '.join(variables)), data_train_model).fit()


# now fitting model to other routes


# In[12]:


sonar_cities = ['ATL', 'CHI', 'DAL', 'LAX', 'PHL', 'SEA']
non_sonar_cities = ['HOU', 'MSP', 'MEM', 'CMH', 'SDF',
                    'CLT', 'DTW', 'STL', 'BWI', 'MKE', 'JAX', 'MSY']

# DAT_cities=sonar_cities.copy()
DAT_cities = list(sonar_cities)
DAT_cities.extend(non_sonar_cities)


DAT_routes = permutations(DAT_cities, 2)

DAT_routes = [x[0]+x[1] for x in DAT_routes]


# In[13]:


days_prediction = 7
holidays = [pd.to_datetime('2019-01-11').date()]


# In[14]:


predictions = pd.DataFrame()

# now need to generate data and predict for each of the routes
for route in DAT_routes:
    # generating data
    data_predict = gen_route_data(route, 'sonar_input', 0)

    for model_index in range(1, len(models_dict)+1):

        # shifting to fit the lag of the variable
        data_predict_lag = data_predict.copy()
        # filling forward in case some values are missing
        data_predict_lag = data_predict_lag.fillna(method='ffill')

        variables = []
        for variable, lag in models_dict['model_specification_'+str(model_index)].items():
            if '_sq' in variable:
                data_predict_lag[variable] = data_predict_lag[variable.replace(
                    '_sq', '')]**2
            data_predict_lag[variable] = data_predict_lag[variable].shift(lag)
            variables.append(variable)

        # meddling with dates
        last_date = data_predict_lag.index[-1]
        data_predict_lag = data_predict_lag.reset_index(drop=True)

        # making future date
        indices = []
        for index in range(1, days_prediction+1):
            new_date = last_date+timedelta(days=index)
            indices.append(new_date)
            month_end = pd.to_datetime(new_date+MonthEnd(1))
            days_to_month_end = (month_end-pd.to_datetime(new_date)).days
            # marking month_end variable
            if days_to_month_end < 5:
                data_predict_lag.loc[index, 'month_end'] = 1

        data_predict_lag = data_predict_lag.iloc[-days_prediction:].copy()
        data_predict_lag.index = indices

        # marking holidays
        for holiday in holidays:
            if holiday in data_predict_lag.index:
                data_predict_lag.loc[holiday, 'TRUK_ORIGIN_slump'] = 1

        data_predict_lag = data_predict_lag.reset_index()
        for date_index in data_predict_lag.index:
            for index in range(-2, 3):
                if (date_index+index) in data_predict_lag.index:
                    data_predict_lag.loc[date_index +
                                         index, 'TRUK_ORIGIN_slump'] = 1

        data_predict_lag = data_predict_lag.set_index('index')

        data_predict.loc[data_predict.index[model_index-1], route] = trained_models['model_specification_' +
                                                                                    str(model_index)].predict(data_predict_lag[variables]).dropna().iloc[-1]
        # .iloc[model_index-1]

    data_predict = data_predict.iloc[:model_index].copy()
    data_predict.index = indices
    predictions = pd.concat([predictions, data_predict[[route]].dropna()])


# In[28]:


predictions = pd.DataFrame(predictions.reset_index().groupby('index').mean())

# # now appending DAT data

# In[96]:


# get existing DAT data
sql_query = '''
select * from dat_input
'''
DAT = get_sql(sql_query)


# In[97]:


# In case DAT is missing, filling in the missing dates and interpolating
idx = pd.date_range(DAT['date'].min(), DAT['date'].max())
DAT = DAT.set_index('date')
DAT = DAT.reindex(idx, fill_value=0)
DAT = DAT.replace(0.00, np.nan)
DAT = DAT.sort_index()


# In[98]:


DAT = DAT.interpolate()
# converting to date to be able to merge it with predictions
DAT.index = [x.date() for x in DAT.index]

# last available DAT day
last_DAT_date = DAT.index.max()
last_DAT_index = DAT['id'].max()


predictions = pd.concat([DAT, predictions], join="inner")
predictions = predictions.sort_index()


# In[102]:


predictions = predictions.loc[last_DAT_date-timedelta(days=6):].copy()


# In[103]:


predictions = predictions.reset_index()
for prediction_index in predictions.index[-7:]:
    for column in [x for x in predictions.columns if x != 'index']:
        predictions.loc[prediction_index, column] = predictions.loc[prediction_index,
                                                                    column] + predictions.loc[prediction_index-7, column]
predictions = predictions.set_index('index')


# In[104]:


predictions = predictions.reset_index().rename(columns={'index': 'date'})
# predictions['prediction_date']=str(datetime.now().year*10000+datetime.now().month*100+datetime.now().day)
predictions['prediction_date'] = (
    datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# In[105]:


predictions = predictions[predictions['date'] > last_DAT_date].copy()
predictions = predictions.reset_index(drop=True)
sql_query = '''
select max(id) from predicted_rates
'''
max_id = get_sql(sql_query)

if max_id.shape[0] > 0:
    max_id = max_id.iloc[0, 0]
else:
    max_id = 0

predictions['id'] = max_id+1+predictions.index

predictions.to_csv(r'predictions.csv', encoding='utf-8')

update_new_rows_mysql(predictions, 'predicted_rates')
