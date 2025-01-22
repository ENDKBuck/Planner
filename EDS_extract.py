# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:15:53 2024

@author: JWB
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta

class EDS_caller():
    
    r"""
    This class creates a model for estimating the cost 
    for supplying the heat- power and ancillirary service demands.
    """
    
    def get_dates(self,DateStart,DateEnd):
        """Get the start and end dates formatted as ISO 8601 without seconds."""
        if DateStart is None or DateEnd is None:        
            now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            DateStart = now - timedelta(days=1)
            DateEnd = DateStart + timedelta(days=1)
        else:
            DateStart = datetime.strptime(DateStart, '%Y-%m-%d %H:%M:%S')
            DateEnd = datetime.strptime(DateEnd, '%Y-%m-%d %H:%M:%S')
            
        Start = DateStart.strftime("%Y-%m-%dT%H:%M")
        End = DateEnd.strftime("%Y-%m-%dT%H:%M")
        
        return Start, End
    
    def fetch_and_process_data(self,url, drop_columns, date_column='HourDK'):
        """Fetch data from the API and process it into a DataFrame."""
        response = requests.get(url)
        data = response.json().get('records', [])
        
        df = pd.DataFrame.from_dict(data)
        df = df.drop(columns=drop_columns)
        
        try:
            df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S')
            df = df.set_index(date_column)
        except:
            df[date_column] = pd.to_datetime(df[date_column], format='ISO8601')
            df = df.set_index(date_column)
        
        return df
    
    def get_Elspot(self,DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
            
        url = f'https://api.energidataservice.dk/dataset/Elspotprices?filter={{"PriceArea":["{area}"]}}&start={Start}&end={End}&sort=HourUTC'
        
        drop_columns = ["HourUTC", "SpotPriceEUR", "PriceArea"]
        
        df = self.fetch_and_process_data(url, drop_columns)
        df = df.rename(columns={"SpotPriceDKK":"Elspot"})
        return df
    
    def get_FCRprices_dk2(self, DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
        
        auctions = ["D-1 early", "D-1 late"]
        
        df = pd.DataFrame()
    
        for auction in auctions:
            url = f'https://api.energidataservice.dk/dataset/FcrNdDK2?filter={{"PriceArea":["{area}"],"AuctionType":["{auction}"]}}&start={Start}&end={End}&sort=HourUTC' 
            drop_columns = ["HourUTC", "PurchasedVolumeLocal", "PurchasedVolumeTotal", "PriceArea", "AuctionType"]
            dftemp = self.fetch_and_process_data(url, drop_columns)
            dftemp = dftemp.pivot(columns='ProductName', values='PriceTotalEUR')
            
            #Euro
            dftemp = dftemp * 7.45
            
            #Renames
            auctionName = auction.split()[-1]
            dftemp = dftemp.rename(columns={"FCR-D ned":"FCR-D_Down_"+auctionName+"_Price",'FCR-D upp':"FCR-D_Up_"+auctionName+"_Price",'FCR-N':"FCR-N_"+auctionName+"_Price"})
            
            df = pd.concat([df, dftemp], axis = 1)
        return df
    
    def get_FCRprices_dk1(self, DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
        
        url = f'https://api.energidataservice.dk/dataset/FcrDK1?&start={Start}&end={End}&sort=HourUTC' 
        drop_columns = ["HourUTC",'FCRdomestic_MW','FCRabroad_MW', 'FCRcross_EUR','FCRcross_DKK','FCRdk_EUR']
        
        df = self.fetch_and_process_data(url, drop_columns)
       
        df = df.rename(columns={"FCRdk_DKK":'FCR_Price'})
            
        return df
    
    def get_FFR(self, DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
        url = f'https://api.energidataservice.dk/dataset/FFRDK2?&start={Start}&end={End}&sort=HourUTC'
        
        drop_columns = ["HourUTC",'FFR_PriceEUR']
        
        df = self.fetch_and_process_data(url, drop_columns)
        df = df.rename(columns={"FFR_PriceDKK":'FFR_Price'})
        
        return df
    
    
    def get_aFRR(self, DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
    
        url = f'https://api.energidataservice.dk/dataset/AfrrReservesNordic?filter={{"PriceArea":["{area}"]}}&start={Start}&end={End}&sort=HourUTC'
        
        drop_columns = ["HourUTC", "aFRR_DownPurchased", "aFRR_UpPurchased", "aFRR_UpCapPriceEUR", "aFRR_DownCapPriceEUR", "PriceArea"]
        
        df = self.fetch_and_process_data(url, drop_columns)
        df = df.rename(columns={"aFRR_DownCapPriceDKK":'aFRR_Down_Price',"aFRR_UpCapPriceDKK":'aFRR_Up_Price'})
        return df
    
    def get_mFRR(self,DateStart, DateEnd, area):
        Start, End = self.get_dates(DateStart,DateEnd)
    
        url = f'https://api.energidataservice.dk/dataset/mFRRCapacityMarket?filter={{"PriceArea":["{area}"]}}&start={Start}&end={End}&sort=HourUTC'
        
        drop_columns = ["HourUTC", "mFRR_DownPurchased", "mFRR_UpPurchased", "mFRR_UpPriceEUR", "mFRR_DownPriceEUR", "PriceArea"]
       
        df = self.fetch_and_process_data(url, drop_columns)
        df = df.rename(columns={"mFRR_DownPriceDKK":'mFRR_Down_Price',"mFRR_UpPriceDKK":'mFRR_Up_Price'})
        return df
    
    def get_tariffs(self, DateStart,DateEnd,DSO,Chargetype):
        Start, End = self.get_dates(DateStart,DateEnd)
        url = f'https://api.energidataservice.dk/dataset/DatahubPricelist?filter={{"ChargeOwner":["{DSO}"],"ChargeTypeCode":["{Chargetype}"]}}&end=now&sort=ValidFrom%20desc&limit=20'
        
        drop_columns = ["GLN_Number", "ChargeType", "ChargeTypeCode", "Description", "TransparentInvoicing", "TaxIndicator", "ResolutionDuration", "VATClass","ChargeOwner","Note"]
        
        df = self.fetch_and_process_data(url, drop_columns, date_column='ValidFrom')
        
        # Renames Price1 to Hour1
        df.columns= df.columns.str.replace('Price','Hour',regex=True)
        
        # Initialize an empty list to store the rows of the new DataFrame
        
        # Convert ValidTo to datetime
        df['ValidTo'] = pd.to_datetime(df['ValidTo'])    
            
        timeseries_data = []
    
        # Iterate through each row in the original DataFrame
        for index, row in df.iterrows():
            current_time = index
            while current_time < row['ValidTo']:
                hour_index = current_time.hour + 1
                hour_value = row[f'Hour{hour_index}']
                timeseries_data.append({
                    'DateTime': current_time,
                    'Value': hour_value
                })
                current_time += pd.Timedelta(hours=1)
        
        # Create the timeseries DataFrame
        timeseries_df = pd.DataFrame(timeseries_data) 
        timeseries_df['DateTime'] = pd.to_datetime(timeseries_df['DateTime'], format='%Y-%m-%d %H:%M:%S')
        timeseries_df = timeseries_df.set_index('DateTime')
        
        # Converts tariffs from DKK/kWh into DKK/MWh
        timeseries_df.Value = timeseries_df.Value*1000
        # Renames to tariff
        timeseries_df = timeseries_df.rename(columns={'Value':'Tariffs'})
        return timeseries_df

    def run_EDS(self, DateStart, DateEnd, area, DSO, Connection_type):
        
        df_elspot = self.get_Elspot(DateStart, DateEnd, area)
        FCRprices_dk2 = self.get_FCRprices_dk2(DateStart, DateEnd,  area) if area =="DK2" else None 
        FCRprices_dk1 = self.get_FCRprices_dk1(DateStart, DateEnd,  area) if area =="DK1" else None
        get_FFR = self.get_FFR(DateStart, DateEnd, area)
        get_aFRR = self.get_aFRR(DateStart, DateEnd, area)
        get_mFRR = self.get_mFRR(DateStart, DateEnd, area)
        get_tariffs = self.get_tariffs(DateStart,DateEnd, DSO, Connection_type)
        
        df_input = pd.concat([df_elspot,FCRprices_dk2,FCRprices_dk1,get_FFR,get_aFRR,get_mFRR,get_tariffs],join='inner',axis = 1)
        
        return df_input