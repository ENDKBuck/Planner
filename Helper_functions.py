# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:15:24 2024

@author: JWB
"""

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests
import json
import random

#%% 

class ticToc():
    r"""
    Class for assesing the programs time consumption
    """
    def __init__(self,ticTocName=''):
        self.tic = time.time()
        self.ticTocName = ticTocName
    
    def toc(self):
                
        print(self.ticTocName+' calculation time: '+'%.3f seconds.'%(time.time()-self.tic))


def linear_regression(subset, x_col, y_col):
    if len(subset) > 1:
        x_values = subset[x_col].values.reshape(-1, 1)
        y_values = subset[y_col].values
    
        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(x_values, y_values)
    
        # Extract coefficients
        a = model.coef_[0]
        b = model.intercept_
    else:
        # If only one row, set coefficients to None
        a = None
        b = None
    
    return a, b


def Heat_input(DateStart,DateEnd,TimeSeries): 
    
    
    # Function for forecast
    def fetch_temperature_forecast(DateStart,DateEnd):
    
        # Convert DateStart and DateEnd to datetime objects
        start_date = datetime.strptime(DateStart, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(DateEnd, "%Y-%m-%d %H:%M:%S")
        
        # Add 3 days to the end date using timedelta
        new_end_date = end_date + timedelta(days=3)
        
        # Convert both dates to ISO 8601 format with 'Z' for UTC
        start_date_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        new_end_date_iso = new_end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Construct the datetime parameter
        datetime_param = f"{start_date_iso}/{new_end_date_iso}"
            
            
        api_key = '7f560113-d22b-4798-8803-4ffdfe8d148d' 
        
         
        # Define the API endpoint and query parameters
        url = 'https://dmigw.govcloud.dk/v1/forecastedr/collections/harmonie_dini_sf/position'
        params = {
            'coords': 'POINT(9.4725 55.4969)',  # Coordinates in WKT format
            'crs': 'crs84',                   # Coordinate reference system
            'parameter-name': 'temperature-0m',  # Parameter to fetch
            'f': 'GeoJSON',                   # Response format
            'datetime':datetime_param, # Example'2025-01-12T00:00:00Z/2025-02-18T00:00:00Z',
            'api-key': api_key                # API key
        }
        
        # Send a GET request to the API
        response = requests.get(url, params=params)
        data = response.json()
        
        features = data.get('features', [])
        
        # Create a DataFrame from the features
        data_temp = pd.DataFrame([
             {'timestamp': feature['properties']['step'],  # Timestamp
             'temperature': feature['properties']['temperature-0m']   # Temperature value
             }
             for feature in features
         ])
        
        
        # Convert timestamp to datetime 
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp']).dt.tz_convert('CET')
        data_temp['timestamp'] = data_temp['timestamp'].dt.tz_localize(None)
        
        
        # Convert temperature from Kelvin to Celsius
        data_temp['temperature'] = data_temp['temperature'] - 273.15
        
        return data_temp
    
    #Function for actual temperature
    def fetch_temperature(DateStart,DateEnd):
    
        # Convert DateStart and DateEnd to datetime objects
        start_date = datetime.strptime(DateStart, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(DateEnd, "%Y-%m-%d %H:%M:%S")
        
        # Add 2 hours to both start and end dates
        start_date_widened = start_date - timedelta(hours=5)  # Subtract 2 hours from start
        end_date_widened = end_date + timedelta(hours=5)      # Add 2 hours to end
        
        # Convert both dates to ISO 8601 format with 'Z' for UTC
        start_date_iso = start_date_widened.strftime("%Y-%m-%dT%H:%M:%SZ")
        new_end_date_iso = end_date_widened.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Construct the datetime parameter
        datetime_param = f"{start_date_iso}/{new_end_date_iso}"
            
                 
        api_key = '7b168805-e8a3-4330-96e3-f9aa799c8420' 
        
    
         
        # Define the API endpoint and query parameters
        url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items?'
        params = {
            'parameterId':'temp_mean_past1h', # Parameter to fetch                   
            'stationId': '06108',  # Station 
            'limit': 20000,                   # Max fetch
            'datetime':datetime_param, # Example'2025-01-12T00:00:00Z/2025-02-18T00:00:00Z',
            'api-key': api_key                # API key
        }
        
        # Send a GET request to the API
        response = requests.get(url, params=params)
        data = response.json()
        
        features = data.get('features', [])
        
        # Create a DataFrame from the features
        data_temp = pd.DataFrame([
             {'timestamp': feature['properties']['observed'],  # Timestamp
             'temperature': feature['properties']['value']   # Temperature value
             }
             for feature in features
         ])
        
        
        # Convert timestamp to datetime 
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp']).dt.tz_convert('CET')
        data_temp['timestamp'] = data_temp['timestamp'].dt.tz_localize(None)
        
        return data_temp
    
    temp = fetch_temperature(DateStart,DateEnd)
    
    # Define constants
    max_demand = 20
    min_demand = 5
    min_temp = -5    # Temperature corresponding to max demand
    max_temp = 25    # Temperature corresponding to min demand
    
    # Calculate demand based on temperature
    temp['Heat_Demand'] = (max_demand - ((temp['temperature'] - min_temp) / (max_temp - min_temp)) * (max_demand - min_demand)).clip(lower=min_demand, upper=max_demand)
    temp.set_index('timestamp', inplace=True)        
            
    #COP
    max_cop = 5
    min_cop = 3
    min_cop_temp = -5    # Temperature corresponding to max demand
    max_cop_temp = 25    # Temperature corresponding to min demand    

    temp['COP'] = (max_cop - ((temp['temperature'] - max_cop_temp) / (min_cop_temp - max_cop_temp)) * (max_cop - min_cop)).clip(lower=min_cop, upper=max_cop)
    temp['cop_factor'] = (-0.0008*temp['temperature']**2 + 0.0176*temp['temperature']+0.966).clip(upper = 1.07)
    TimeSeries = TimeSeries.merge(temp, how='left', left_index=True, right_index=True)
    
    return TimeSeries   
    
        
        
def PQL_builder(df, tSet):    

    #Get MTU
    MTU = pd.Timedelta(tSet[1]-tSet[0]).seconds/60
    
    # Sort the DataFrame based on the absolute values of 'P'
    df['abs_P'] = np.abs(df['P'])
    df = df.sort_values(by='abs_P', ascending=False).drop(columns='abs_P')   
 
    lineSegment = {}

    for group, subset in df.groupby('Line segment'):
        segment_info = {}
       
        for col in ['P','E','Q','F']:
            colMax = subset[col].max()
            colMin = subset[col].min()
            segment_info[f'{col.lower()}Max'] = colMax
            segment_info[f'{col.lower()}Min'] = colMin
        
        #Batteries
        segment_info['a_pe'], segment_info['b_pe'] = linear_regression(subset, 'P', 'E')
        
        #PQ Units
        segment_info['a_qp'], segment_info['b_qp'] = linear_regression(subset, 'Q', 'P')
        segment_info['a_qf'], segment_info['b_qf'] = linear_regression(subset, 'Q', 'F')
        
        
        #Adds if consumption or production
        segment_info['direction'] = (-1 if segment_info['pMin'] <0 else 1)
        segment_info['node'] = subset.Node[0]
        
        # Add remaining columns to segment_info
        for col in subset.columns:
            if col not in ['P', 'Q', 'F','Node']:
                segment_info[col] = subset[col].mean()
                
        # Calcualtes Ramping time to absolute MTU's
        for col in ['RampUp', 'RampDown']:
            segment_info[f'{col}'] = segment_info[col] * MTU
         
        lineSegment[group] = segment_info

    
    df = pd.DataFrame.from_dict(lineSegment, orient='index').T
        
    return df 



def Bid_builder(df, tSet):    
    
    #Get MTU
    MTU = pd.Timedelta(tSet[1]-tSet[0]).seconds/60
    
    df_bid = pd.DataFrame()

    # Number of bins and bin edges for Elspot
    num_bins = 10
    elspot_bins = np.round(np.linspace(-500, 5000, num_bins),0)
    
    
    # Create a DataFrame for binning Elspot values
    df_bid['Cut'] = pd.cut(df['Elspot'], bins=elspot_bins)
    df_bid['Bin'] = pd.cut(df['Elspot'], bins=elspot_bins, labels=False)
    
    # Initialize a DataFrame to store the budget matrix
    df_bud_matrix = pd.DataFrame(index=tSet, columns=elspot_bins)
    
    # Sums netposition
    pNet = df.filter(regex=r'\.P$').sum(axis=1)
    
    # Fill the budget matrix with the corresponding Battery.P values
    for t in tSet:
        bin_index = df_bid.loc[t, 'Bin']
        bidVolume = np.round(pNet[t],1)
        
        df_bud_matrix.loc[t, elspot_bins[bin_index]] = bidVolume
        
        if bidVolume < 0:
            
            # Fill values in columns prior to the bin_index
            for col_index in range(bin_index + 1):
                df_bud_matrix.loc[t, elspot_bins[col_index]] = bidVolume
       
        else:
           # Fill values in columns post the bin_index
           for col_index in range(bin_index, len(elspot_bins)):
               df_bud_matrix.loc[t, elspot_bins[col_index]] = bidVolume
    
    # Optional: Reset index and fill NaN values if needed
    # df_bud_matrix.reset_index(inplace=True)
    df_bud_matrix.fillna(0, inplace=True)
        
    
        
    return df_bud_matrix 



def clearing_algo_average_profits(df,TimeSeries, tSet,aSet,ancServices,uSet):    
        
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 500
    
    #Chronological order
    ClearOrder = list((ancServices[ancServices['Active'] > 0].sort_values(by='Clearing order').index))
    
    cap_flex = pqlData["Battery"].loc[['pMax','pMin']].abs().max().max()
    
    #Insert bid price
    bids = range(0,2000,50)

    
    df_res = pd.DataFrame(columns=['bids', 'profit'])
    
    
    for bidprice in bids:
        
        #Empty dataframe to show clearing
        df_accepted = pd.DataFrame(index=tSet)
        df_Up = pd.DataFrame(index=tSet)
        df_Dwn = pd.DataFrame(index=tSet)

        
        # initiate the 
        df_Up['Flex_MW'] = cap_flex #- df['Battery.P']   
        df_Up['remain_MW'] = cap_flex #- df['Battery.P']
        df_Up['profit'] = 0
                                           
        df_Dwn['Flex_MW'] = cap_flex #+ df['Battery.P']     
        df_Dwn['remain_MW'] = cap_flex #+ df['Battery.P']   
        df_Dwn['profit'] = 0
        
        for ancs in ClearOrder:
            df_accepted[ancs+'_MW'] = (TimeSeries[ancs + '_Price'] > bidprice).astype(int)
            
            if ancs == "FCR-N_early":
                cap = (df_Up['remain_MW'].combine(df_Dwn['remain_MW'], min))/2
                
            elif ancs == "FCR-N_late":
                cap = ((df_Up['remain_MW'] - df_Up['FCR-N_early_MW']).combine((df_Dwn['remain_MW'] - df_Dwn['FCR-N_early_MW']), lambda x, y: np.minimum(x, y))).clip(lower=0).min() /2
                
            else:
                cap = cap_flex
    
            if ancServices.loc[ancs]["Up"] == 1:
                df_Up[ancs + '_MW'] = df_accepted[ancs + '_MW'] * ancServices["Battery"][ancs] * (cap if "FCR-N" in ancs else cap/(1+ancServices["NEM"][ancs]))
                df_Up[ancs + '_MW'] = df_Up[ancs + '_MW'].clip(upper=df_Up['remain_MW'])
                df_Up['remain_MW'] -= df_Up[ancs + '_MW']
                df_Up['profit'] += df_Up[ancs + '_MW'] * (TimeSeries[ancs + '_Price'] - ancServices['LoadFactor'][ancs] * pqlData['Battery'].loc['Ancillary service cost'].max())
    
            if ancServices.loc[ancs]["Dwn"] == 1:
                df_Dwn[ancs + '_MW'] = df_accepted[ancs + '_MW'] * ancServices["Battery"][ancs]  * (cap if "FCR-N" in ancs else cap/(1+ancServices["NEM"][ancs]))
                df_Dwn[ancs + '_MW'] = df_Dwn[ancs + '_MW'].clip(upper=df_Dwn['remain_MW'])
                df_Dwn['remain_MW'] -= df_Dwn[ancs + '_MW']
                df_Dwn['profit'] += df_Dwn[ancs + '_MW'] * (TimeSeries[ancs + '_Price'] - ancServices['LoadFactor'][ancs] * pqlData['Battery'].loc['Ancillary service cost'].max()) * (0 if "FCR-N" in ancs else 1)
            
        total_profit = int(df_Dwn['profit'].sum() + df_Up['profit'].sum())
        print('profit:', total_profit)
        df_res = pd.concat([df_res, pd.DataFrame({'bids': [bidprice], 'profit': [int(total_profit)]})])
    
    df_res.plot(x='bids',y='profit',kind="area",stacked = False, ylabel ="Profit [DKK]",xlabel ="Bid price [DKK/MW/h]", title="profit per average bid")





def clearing_algo_hour(df,TimeSeries, tSet,aSet,ancServices,uSet):    
    
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 500
    
    #Chronological order
    ClearOrder = list((ancServices[ancServices['Active'] > 0].sort_values(by='Clearing order').index))
    
    cap_flex = pqlData["Battery"].loc[['pMax','pMin']].abs().max().max()
    
    
    #Bidmatrix per hour - max prices
    revenue =  [TimeSeries[anc + '_Price'] - ancServices['LoadFactor'][anc] * pqlData['Battery'].loc['Ancillary service cost'].max() for anc in aSet]
    bidprice = pd.concat(revenue, axis=1).max(axis=1)
    
    # Confidence interval
    confidence = np.linspace(0,1,11)
    
    # Output matrix
    df_res = pd.DataFrame(columns=['confidence', 'profit'])
    
    for c in confidence:
        c= 0.7
        #Empty dataframe to show clearing
        df_accepted = pd.DataFrame(index=tSet)
        df_Up = pd.DataFrame(index=tSet)
        df_Dwn = pd.DataFrame(index=tSet)
    
        
        # initiate the 
        df_Up['Flex_MW'] = cap_flex #- df['Battery.P']   
        df_Up['remain_MW'] = cap_flex #- df['Battery.P']
        df_Up['NEM_MW'] = 0 #+ df['Battery.P']   
        df_Up['profit'] = 0
                                           
        df_Dwn['Flex_MW'] = cap_flex #+ df['Battery.P']     
        df_Dwn['remain_MW'] = cap_flex #+ df['Battery.P']   
        df_Dwn['NEM_MW'] = 0 #+ df['Battery.P']   
        df_Dwn['profit'] = 0
        
        for ancs in ClearOrder:
            if ancs == "FFR":
                break
            ancs = "FCR-N_late"
            c=0
            df_accepted[ancs+'_MW'] = (TimeSeries[ancs + '_Price'] - ancServices['LoadFactor'][ancs] * pqlData['Battery'].loc['Ancillary service cost'].max() >= bidprice*c).astype(int)
            
            if ancs == "FCR-N_early":
                cap = (df_Up['remain_MW'].combine(df_Dwn['remain_MW'], min))/2
                
            elif ancs == "FCR-N_late":
                cap = ((df_Up['remain_MW'] - df_Up['FCR-N_early_MW']).combine((df_Dwn['remain_MW'] - df_Dwn['FCR-N_early_MW']), lambda x, y: np.minimum(x, y))).clip(lower=0).min() /2
                
            else:
                cap = cap_flex
    
            if ancServices.loc[ancs]["Up"] == 1:
                df_Up[ancs + '_MW'] = df_accepted[ancs + '_MW'] * ancServices["Battery"][ancs] * cap
                df_Up[ancs + '_MW'] = df_Up[ancs + '_MW'].clip(upper=df_Up['remain_MW'])
                df_Up['remain_MW'] -= df_Up[ancs + '_MW'] * (1+ancServices["NEM"][ancs])
                df_Up['NEM_MW'] += df_Up[ancs + '_MW'] * ancServices["NEM"][ancs]
                df_Up['profit'] += df_Up[ancs + '_MW'] * (TimeSeries[ancs + '_Price'] - ancServices['LoadFactor'][ancs] * pqlData['Battery'].loc['Ancillary service cost'].max())
    
            if ancServices.loc[ancs]["Dwn"] == 1:
                df_Dwn[ancs + '_MW'] = df_accepted[ancs + '_MW'] * ancServices["Battery"][ancs] * cap
                df_Dwn[ancs + '_MW'] = df_Dwn[ancs + '_MW'].clip(upper=df_Dwn['remain_MW'])
                df_Dwn['remain_MW'] -= df_Dwn[ancs + '_MW']  * (1+ancServices["NEM"][ancs])
                df_Dwn['NEM_MW'] += df_Dwn[ancs + '_MW'] * ancServices["NEM"][ancs]
                df_Dwn['profit'] += df_Dwn[ancs + '_MW'] * (TimeSeries[ancs + '_Price'] - ancServices['LoadFactor'][ancs] * pqlData['Battery'].loc['Ancillary service cost'].max()) * (0 if "FCR-N" in ancs else 1)
            
        total_profit = int(df_Dwn['profit'].sum() + df_Up['profit'].sum())
        print('profit:', total_profit)
        df_res = pd.concat([df_res, pd.DataFrame({'confidence': [c], 'profit': [int(total_profit)]})])

    df_res.plot(x='confidence',y='profit',kind="area",stacked = False, ylabel ="Profit [DKK]",xlabel ="Confidence ", title="profit with different confidence levels",figsize=(8, 4))
    
    return df 

