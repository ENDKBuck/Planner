# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 07:45:06 2024

@author: JWB
"""

# Standard modules
import pandas as pd
from datetime import timedelta 
import pickle

#Installed modules
import xlwings as xw

#Loads function from scripts in folder
from Planner import *
from Planner_MathOpt import *
from Helper_functions import *
from EDS_extract import *
from plotter import *


#Reads Excel
wb = xw.Book('Dispatcher_all.xlsm')             

# Empty dataframe
df = pd.DataFrame()
pqlData = {}

# Import data from Excel 
Main            = wb.sheets['Main'].range('A1').options(pd.DataFrame, expand='table').value
ancServices     = wb.sheets['Ancillary services'].range('A1').options(pd.DataFrame, expand='table').value
BatStorage     = wb.sheets['Battery Storage'].range('A1').options(pd.DataFrame, expand='table').value
qStorage        = wb.sheets['Storage'].range('A1').options(pd.DataFrame, expand='table').value
Setup           = (wb.sheets['Setup'].range('A1').options(pd.DataFrame, expand='table').value).drop('Unit',axis=1)
soldCap         = wb.sheets['Sold Capacity'].range('A1').options(pd.DataFrame, expand='table').value


#Input for EDS data extraction
DateStart = str(Main.Input.DateStart) #'2024-05-01 00:00:00' #None (last 24 hours)
DateEnd = str(Main.Input.DateEnd) #'2024-05-14 00:00:00' #None (last 24 hours)
DSO = Setup.loc["DSO",:][0]
Connection_type = Setup.loc["Connection_type",:][0]
area  = Setup.loc["area",:][0]

# Call the functions
TimeSeries = EDS_caller().run_EDS(DateStart, DateEnd, area, DSO, Connection_type)
TimeSeries = Heat_input(DateStart,DateEnd, TimeSeries)

wb.sheets['TimeSeries'].range('A1').value = TimeSeries


# Creates Sets
uSet            = list(Setup.columns)
aSet            = list(ancServices[ancServices['Active'] > 0].index)
tSet            = list(TimeSeries.index)

# Creates batches (optimization loops)
FS              = int(Main.Input.Foresight - Main.Input['Perfect Foresight'])
tLen            = int(Main.Input['Perfect Foresight'])*24
tLoop           = list([tSet[i:i+tLen] for i in range(0, len(tSet), tLen)])
end             = len(tLoop)

# Creating Unit data
for u in uSet:    
    pqlData[u]  = PQL_builder(wb.sheets[u].range('J1').options(pd.DataFrame, expand='table').value,tSet)
    TimeSeries[u] = TimeSeries['cop_factor'] if u == "HP" else 1
    
# Max solve time (min)
maxTime= int(Main.Input.MaxTime)


# Starts a clock for optimization
tic = ticToc('Optimize')

# Initiates loop with sliding timeframe
for num,t in enumerate(tLoop):
    tBatch =  pd.date_range(start=t[0], end=t[-1] + timedelta(days=FS), freq='h')    
    
    # When last iteration - get last values 
    if num == len(tLoop)-1:   
        tBatch =  pd.date_range(start=t[0], end=t[-1], freq='h')
    
    print(tBatch[0],'->',tBatch[-1])
    
    loop_timeseries = TimeSeries[tBatch[0]:tBatch[-1]]
    loop_soldCap    = soldCap[tBatch[0]:tBatch[-1]]
    
    # Run_Planner [CBC Solver] // run_plan_optimization [MathOpt HIGHS]
    df_temp = run_plan_optimization(loop_timeseries,
                      BatStorage,
                      qStorage,
                      Setup,
                      pqlData,
                      ancServices,
                      df,
                      t,
                      uSet,
                      aSet,
                      maxTime,
                      num,
                      end,
                      loop_soldCap)


    df = pd.concat([df, df_temp], ignore_index=False)

print('\nTotal costs, MDKK =', df.TotalCost.sum())

# Ends Clock
tic.toc()   

# Stores as python-pickle
data = (df, TimeSeries, aSet,ancServices, Setup,BatStorage,qStorage,pqlData,loop_soldCap,uSet)
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)


# Data to Excel        
ws = wb.sheets["Results"]
ws.name = "Results"
ws.clear_contents()
ws.range("A1").value = df

# DA BID to excel
ws = wb.sheets["DA Bidder"]
df_bud_matrix = Bid_builder(df, tSet)
ws.range("A16").value = df_bud_matrix

# creates html figures
Plotter(df, aSet, TimeSeries, Setup)
Plotter_DH(df, aSet, TimeSeries, Setup)
Plotter_battery(df,ancServices, aSet, TimeSeries, Setup)
