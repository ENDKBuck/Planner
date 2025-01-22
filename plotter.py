# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:20:19 2024

@author: JWB
"""


import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 500



def Plotter_DH(df,aSet,TimeSeries,Setup):   
    

    x   = TimeSeries.index
    
     
    fig1 = make_subplots(rows = 2, cols =1, row_width=[0.3, 0.7], specs=[[{"secondary_y": True}],[{"secondary_y": True}]],subplot_titles=["Heat Production","Power production"],vertical_spacing = 0.1)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
     
    ActiveUnits = Setup.columns
    plotList = ActiveUnits
        
        
    for u in ActiveUnits:
        fig1.add_trace(go.Scatter(x=x, y=df[u+'.P'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'),line_color=Setup[u].plotColor, name=u+'-P',legendgroup='P'+u, stackgroup='Two'),row=2,col=1) 
        if u != "Battery":
            fig1.add_trace(go.Scatter(x=x, y=df[u+'.Q'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'),line_color=Setup[u].plotColor, name=u+'.Heat',legendgroup='Q'+u, stackgroup='Two'),row=1,col=1) 
            fig2.add_trace(go.Scatter(x=df[u+'.Q'], y=df[u+'.P'], hoverinfo='x+y', mode='markers',opacity=0.9, name='PQ-'+u,marker=dict(color=Setup[u].plotColor),legendgroup='group'+u),row=1,col=1) 
    
            
    fig1.add_trace(go.Scatter(x=x, y=df['Storage'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Storage',legendgroup='Storage',line_color='blue')) 
    fig1.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',legendgroup='Elspot',line_color='black'),secondary_y=True,row=2,col=1)  
    fig1.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',legendgroup='Elspot',line_color='black'),secondary_y=True,row=1,col=1)  
    fig1.add_trace(go.Scatter(x=x, y=df['Heat Demand'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Heat Demand',legendgroup='Heat Demand',line_color='red')) 
    
    

    fig1.update_xaxes(title_text='Time')
    fig1.update_yaxes(title_text='MW')
    fig1.update_yaxes(title_text="kr./MWh", secondary_y=True)
    
    fig1.write_html('EnergyProd.html', auto_open=True)
    fig2.write_html('PQ-diagram.html', auto_open=True)    




def Plotter(df,aSet,TimeSeries,Setup):   
    

    x   = TimeSeries.index
    
    fig = make_subplots(
    rows=len(aSet),  # Number of rows equals the length of aSet
    cols=1,
    specs=[[{"secondary_y": True}] for _ in range(len(aSet))],  # Create specs dynamically for each row
    subplot_titles=aSet,
    vertical_spacing=0.05)
     
    ActiveUnits = Setup.columns
    plotList = ActiveUnits
        
        
    for u in ActiveUnits:
        for num,ancs in enumerate(aSet):
            fig.add_trace(go.Scatter(x=x, y=df[u+'.'+ancs], hoverinfo='x+y',mode='lines',line=dict(shape='hv'),line_color=Setup[u].plotColor, name=u+'-'+ancs,legendgroup='P'+u,legendgrouptitle_text="Sold Ancillary services", stackgroup='one'),row=num+1,col=1) 
    
    
    for num,ancs in enumerate(aSet):        
            fig.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',line_color='black',legendgroup='Elspot',showlegend = True if num == 0 else False),secondary_y=True,row=num+1,col=1)    
            fig.add_trace(go.Scatter(x=x, y=TimeSeries[ancs+'_Price'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name=ancs+"-Price",line_color='blue',legendgroup='Elspot',legendgrouptitle_text="Prices"),secondary_y=True,row=num+1,col=1)    
             
    
    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text='MW/h')
    fig.update_yaxes(title_text="kr./MW/h", secondary_y=True)
    fig.update_traces(marker_size=10)
    fig.update_layout(legend=dict(groupclick="toggleitem"),autosize=True,height=2000)
    fig.write_html('Ancillary services.html', auto_open=True)
    

def Plotter_battery(df,ancServices,aSet,TimeSeries,Setup):   
    

    x   = TimeSeries.index
    
    fig = make_subplots(rows = 2, cols = 1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]],subplot_titles=["Power Balance","Ancillary services"])
    
    fig.add_trace(go.Scatter(x=x, y=df['Battery.P'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'),line_color=Setup["Battery"].plotColor, name='Battery',legendgroup='Battery', stackgroup='one',legendgrouptitle_text="Operation"),row=1,col=1) 
    fig.add_trace(go.Scatter(x=x, y=df['Battery.SoC'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), line_color='cornsilk', name='SoC',legendgroup='Battery', stackgroup='two',legendgrouptitle_text="Operation"),row=1,col=1)   
    fig.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',legendgroup='Elspot',line_color='black',legendgrouptitle_text="Operation"),secondary_y=True,row=1,col=1)          
    
    for ancs in aSet:
        plotColor =  ancServices['plotColor'][ancs]
        
        if (ancServices.loc[ancs]['Up'] == 1) & (ancServices.loc[ancs]['Dwn'] == 1):
            fig.add_trace(go.Scatter(x=x, y=df['Battery.'+ancs], hoverinfo='x+y',mode='lines',line=dict(shape='hv', width=0), line_color=plotColor,legendgroup = "Symmetric", legendgrouptitle_text="Ancillary Services-"+"Symmetrical", name='Battery-'+ancs, stackgroup="Up"),row=2,col=1) 
            fig.add_trace(go.Scatter(x=x, y=-df['Battery.'+ancs], hoverinfo='x+y',mode='lines',line=dict(shape='hv',width=0), line_color=plotColor,legendgroup = "Symmetric", legendgrouptitle_text="Ancillary Services-"+"Symmetrical", name='Battery-'+ancs, stackgroup='Down'),row=2,col=1) 
        
        else:
            direction = ("Up" if ancServices.loc[ancs]['Up'] == 1 else "Down")
            sign = (1 if ancServices.loc[ancs]['Up'] == 1 else -1)
            fig.add_trace(go.Scatter(x=x, y=df['Battery.'+ancs]*sign, hoverinfo='all',mode='lines',line=dict(shape='hv',width=0),line_color=plotColor, legendgroup = direction, legendgrouptitle_text="Ancillary Services-"+direction, name='Battery-'+ancs, stackgroup=direction),row=2,col=1) 
        
    fig.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',legendgroup='Elspot',line_color='black',showlegend=False),secondary_y=True,row=2,col=1)          
    
    
    # Adding custom y-axis labels
    fig.update_yaxes(title_text="Power [MW]", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Elspot [DKK/MWh]", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Power [MW]", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Elspot [DKK/MWh]", secondary_y=True, row=2, col=1)
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    
    fig.update_layout(hovermode="x")
    
    fig.write_html('Battery.html', auto_open=True)   
    fig.show() 
    
    
def plot_prices(TimeSeries,aSet):   
    

    x   = TimeSeries.index
        
    fig = go.Figure() 
    
    fig.add_trace(go.Scatter(x=x, y=df['Elspot'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'), name='Elspot',line_color='black',legendgroup='Prices',legendgrouptitle_text="Prices",showlegend = True))    
    
    for ancs in aSet: 
        plotColor =  ancServices['plotColor'][ancs]
        fig.add_trace(go.Scatter(x=x, y=TimeSeries[ancs+'_Price'], hoverinfo='x+y',mode='lines',line=dict(shape='hv'),line_color=plotColor, name=ancs+"-Price",legendgroup='Prices',legendgrouptitle_text="Prices"))    
    

    # Highlight the highest prices among ancillary services per hour with a transparent line  
    max_prices = TimeSeries[[anc+'_Price' for anc in aSet]].max(axis=1)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=max_prices,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.2)', width=12,shape='hv'),  # Red line with transparency
        name='Max Ancillary Price',
        showlegend=True,
        hoverinfo='skip'))
    
    # revenue =  [TimeSeries[anc + '_Price'] - ancServices['LoadFactor'][anc] * pqlData['Battery'].loc['Ancillary service cost'].max() for anc in aSet]
    # bidprice = pd.concat(revenue, axis=1).max(axis=1)    
    
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=bidprice,
    #     mode='lines',
    #     line=dict(color='rgba(11, 156, 49, 0.2)', width=12,shape='hv'),  # Red line with transparency
    #     name='Maximal Revenue',
    #     stackgroup='two',
    #     showlegend=True,
    #     hoverinfo='skip'))

    
    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text="kr./MW/h")
    fig.update_traces(marker_size=10)
    fig.update_layout(template="ggplot2",legend=dict(groupclick="toggleitem"))
    fig.write_html('prices.html', auto_open=True)