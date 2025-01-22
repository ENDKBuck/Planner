# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 07:13:23 2021

@author: JWB
"""

from ortools.linear_solver import pywraplp
import pandas as pd
import itertools
from Helper_functions import *


def Run_Planner(loop_timeseries,
                 BatStorage,
                 qStorage,
                 Setup,
                 pqlData,
                 ancServices,
                 df,
                 tSet,
                 uSet,
                 aSet,
                 maxTime,
                 num,
                 end,
                 loop_soldCap):
   
    #Calls and executes the Planner optimization
    optimizer = Planner(tSet, uSet, aSet, loop_timeseries, qStorage, BatStorage, Setup, pqlData, ancServices, df,maxTime, num, end, loop_soldCap)
    
    
    #Fetches the results after the optimization
    df = optimizer.extract_results_from_solver()

    
    ## Examples: Call results from the stored variables from the optimizer:
        
    #optimizer.model.Objective().Value()
    #optimizer.var.Storage[t].solution_value()
    
    return df

class expando(object):
    r"""
        A small class which can have attributes set
    """
    pass


class Planner():
    r"""
    This class creates a model for estimating the cost 
    for supplying the heat- power and ancillirary service demands.
    """
    
    def __init__(self,tSet, uSet, aSet, loop_timeseries, qStorage, BatStorage, Setup, pqlData, ancServices, df,maxTime, num, end, loop_soldCap):
        r"""
        Creates the initial structure for data, variables etc.
        """
        
        self.data   = expando()
        self.var    = expando()
        self.sets   = expando()
        self.U      = {}
        
        # Loads the dataset into dictionary
        self.load_data(tSet,uSet,aSet,loop_timeseries,BatStorage,qStorage,Setup,pqlData,ancServices,df,num,end,loop_soldCap)
        
        #Creates the OR-Tool pywraplp solver
        self.model = pywraplp.Solver('Dispatcher', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)     #  https://google.github.io/or-tools/python/ortools/linear_solver/pywraplp.html
        
        #Creates variables,constraints and objective
        self.set_up_model_variables_and_constraints()
       
        #Call and runs the model
        self.optimize(maxTime)

        
        
        
    def optimize(self,maxTime):
        r"""
        Runs the optimization
        """
        
        #Adds a maximum solver time in ms
        self.model.SetTimeLimit(1000*60*maxTime)
        
        #Solves the model
        self.modelStatus = self.model.Solve()
    
    
        # Check model status
        if self.modelStatus == pywraplp.Solver.OPTIMAL or self.modelStatus == pywraplp.Solver.FEASIBLE:
            print('Objective value, MDKK =', self.model.Objective().Value())    
            print('Relative gap % =', (self.model.Objective().Value()-self.model.Objective().BestBound())/self.model.Objective().BestBound()*100)
         
        else:
            print('\nThe problem does not have an optimal solution.')  
    
            
    def load_data(self,tSet,uSet,aSet,loop_timeSeries,BatStorage,qStorage,Setup,pqlData,ancServices,df,num,end,loop_soldCap):
        r"""
        Stores the dataset in to an organised dictionary for fast fetching
        """
        
        # Input Data
        self.data.__dict__.update({
            "timeSeries": loop_timeSeries.to_dict(),
            "BatStorage": BatStorage,
            "qStorage": qStorage,
            "Setup": Setup.to_dict(),
            "pqlData": pqlData,
            "ancService": ancServices.to_dict(),
            "df": df,
            "batch": num,
            "batchEnd": end,
            "soldCapacity": loop_soldCap
        })
        
        #Sets to be looped
        self.sets.__dict__.update({
            "t": tSet,  # Timestep
            "u": uSet,  # Unit
            "a": aSet,   # Each ancillary service
            "plants": Setup.columns[Setup.loc['asset type'] == 'plant'].tolist(),   # Thermal plants  
            })

    
        
    def set_up_model_variables_and_constraints(self):
        r"""
        Define variables, constraints, and objectives for multiple timesteps for the optimization problem.
        """

        m = self.model
        
        #Loops units to create unit variables & constraints
        for unit in self.sets.u:
            self.set_up_unit_variables_and_constraints(m, unit,self.data.Setup[unit],self.data.timeSeries,self.data.pqlData[unit],self.data.ancService[unit],self.data.df)
            
            #Creates battery related constraints if units is a battery
            if unit == "Battery":
                self.set_up_battery(m, self.data.timeSeries, self.data.BatStorage, self.data.df, self.data.batch, self.data.batchEnd)
        
        
        #Creates power related constraints         
        self.set_up_powerBalance(m,self.data.timeSeries,self.data.BatStorage,self.data.df,self.data.batch,self.data.batchEnd)
        
        #Creates heat/hydrogen or other primary purpose variables and constraints
        self.set_up_energyBalance(m,self.data.timeSeries,self.data.qStorage,self.data.df,self.data.batch,self.data.batchEnd)
        
        #Creates the objective for the model
        self.set_up_objective_function(m)
        
        
    def set_up_unit_variables_and_constraints(self,m,unit,Setup,timeSeries,pqlData,ancService,df):
        print('Creating -', unit)

        r"""
        This function creates all the unit variables and calls the Unit-constraint function
        to create the associated unit constraints.
        """
        
        #Empty data sockets for fast data storage
        self.U[unit]      = expando()
        self.U[unit].var  = expando()         
        
        #Calcualtions and lookups for speed performance
        loadPoints = {   
            ('pAbs'): (pqlData.loc[['pMax','pMin']].abs().max().max() + (pqlData.loc['qMax'].max() * 1/pqlData.loc['Cv-line'].max() if pqlData.loc['Cv-line'].max() > 0 else 0)),
            ('pMin', 'lb'): min(pqlData.loc['pMin'].min(), 0),  
            ('pMax', 'ub'): (max(pqlData.loc['pMax'].max(), 0) +  (pqlData.loc['qMax'].max() * 1/pqlData.loc['Cv-line'].max() if pqlData.loc['Cv-line'].max() > 0 else 0)),
            ('eMin', 'lb'): min(pqlData.loc['pMin'].min(), 0) if unit == "Battery" else None,  
            ('eMax', 'ub'): max(pqlData.loc['pMax'].max(), 0) if unit == "Battery" else None,
            ('qMax', 'ub'): (pqlData.loc['qMax'].max() + pqlData.loc['pMax'].max() * pqlData.loc['Bypass'].max()) if unit != "Battery" else None,  
            ('fMax', 'ub'): pqlData.loc['fMax'].max() if unit != "Battery" else None
            }


        ## -------- Build variables --------   
        self.U[unit].var.__dict__.update({
            "totalCosts": {},
            "pCosts": {},
            "qCosts": {},
            "fCosts": {},
            "pNet": {},
            "qNet": {},
            "eNet": {},
            "fNet": {},
            "ASRevenue": {},
            "ancService": {},
            "on": {},
            "start": {},
            "PQLine": {},
            "qLine": {},
            "eLine": {},
            "pLine": {},
            "fLine": {},
            "Bypass": {},
            "Condens": {},
            "pPOT": {}
        })
            
        ##  -------- Hourly variables -------- 
        for t in self.sets.t:                              
            
            # Definitions for fast calcualtion
            Unit_avaiablity = timeSeries[unit][t]
            inf  = m.infinity()
            unit_var = self.U[unit].var
            
            
            #Cost related variables
            unit_var.totalCosts[t]  = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.totalCosts.t{t}")          
            unit_var.ASRevenue[t]   = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.ASRevenue.t{t}")
            unit_var.pCosts[t]      = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.pCosts.t{t}")                       
            unit_var.qCosts[t]      = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.qCosts.t{t}")                   
            unit_var.fCosts[t]      = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.fCosts.t{t}") 

            unit_var.on[t]         = m.IntVar(lb=0,ub= 1, name=f"{unit}.var.on.t{t}")    
            unit_var.start[t]      = m.IntVar(lb=0,ub= 1, name=f"{unit}.var.start.t{t}")    
            
            unit_var.pNet[t] = m.NumVar(lb=loadPoints[('pMin','lb')] * Unit_avaiablity,ub=loadPoints[('pMax','ub')] * Unit_avaiablity, 
            name=f"{unit}.var.pNet.t{t}")


            if unit == "Battery":
                unit_var.eNet[t]        = m.NumVar(lb=loadPoints[('pMin','lb')] * Unit_avaiablity, ub = loadPoints[('eMax','ub')] * Unit_avaiablity, 
                name=f"{unit}.var.eNet.t{t}")
                
            else:
                unit_var.qNet[t]        = m.NumVar(lb= 0, ub= loadPoints[('qMax','ub')] * Unit_avaiablity, 
                name=f"{unit}.var.qNet.t{t}")

                unit_var.fNet[t]        = m.NumVar(lb= 0,ub= loadPoints[('fMax','ub')] * Unit_avaiablity, 
                name=f"{unit}.var.fNet.t{t}")

                unit_var.Bypass[t]      = m.NumVar(lb= 0, ub= loadPoints[('pMax','ub')] * pqlData.loc['Bypass'].max(), 
                name=f"{unit}.var.Bypass.t{t}")
    
                unit_var.Condens[t]     = m.NumVar(lb= 0, ub= loadPoints[('qMax','ub')] * pqlData.loc['Cv-line'].max(), 
                name=f"{unit}.var.Condens.t{t}")
                               

            ## --- Loops for ancillary service
            unit_var.ancService[t]   = {}
            
            for AncS in self.sets.a:
                unit_var.ancService[t][AncS]         = m.NumVar(lb=0, ub= loadPoints[('pAbs')]  * ancService[AncS] * Unit_avaiablity, 
                name=f"{unit}.var.ancService.t{t}.{AncS}")            
                                 
        
        ##  -------- PQL variables --------             
        for t in self.sets.t:
            unit_var.PQLine[t]   = {}
            unit_var.pLine[t]    = {}
            unit_var.qLine[t]    = {}
            unit_var.eLine[t]    = {}
            unit_var.fLine[t]    = {}
            unit_var.pPOT[t]     = {}
            
            if unit == "Battery":
                for pql in pqlData:
                    unit_var.PQLine[t][pql]   = m.IntVar(lb=0, ub=1, name=f"{unit}.var.PQLine.t{t}.{pql}")
                    unit_var.pLine[t][pql]    = m.NumVar(lb= min(0,min(pqlData[pql]['pMax'],pqlData[pql]['pMin'])), ub= max(0,pqlData[pql]['pMax']),  name=f"{unit}.var.pLine.t{t}.{pql}")  
                    unit_var.pPOT[t][pql]      = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.pPOT.t{t}.{pql}") 
            else:
                for pql in pqlData:
                    
                    pql_data = pqlData[pql]
                    unit_var.PQLine[t][pql]   = m.IntVar(lb=0, ub=1, name=f"{unit}.var.PQLine.t{t}.{pql}")
                    unit_var.pLine[t][pql]    = m.NumVar(lb= min(0,min(pql_data['pMax'],pql_data['pMin'])), ub= max(0,pql_data['pMax']),  name=f"{unit}.var.pLine.t{t}.{pql}") # max(0,pqlData[pql]['pMax']),  
                    unit_var.qLine[t][pql]    = m.NumVar(lb= 0, ub= pql_data['qMax'],  name=f"{unit}.var.qLine.t{t}.{pql}") 
                    unit_var.fLine[t][pql]    = m.NumVar(lb= 0, ub= pql_data['fMax'],  name=f"{unit}.var.fLine.t{t}.{pql}") 
                    unit_var.pPOT[t][pql]      = m.NumVar(lb=-inf,ub= inf, name=f"{unit}.var.pPOT.t{t}.{pql}")              

       
         #Calls the set_up_unit_constraints to create constraints
        self.set_up_unit_constraints(m,unit,Setup,timeSeries,pqlData,ancService,df,loadPoints)    

    def set_up_unit_constraints(self,m,unit,Setup,timeSeries,pqlData,ancService,df,loadPoints):        
        """
        Function called from set_up_unit_variables_and_constraints to create the unit constraints.
        Note the constraints are not saved, but can be called via the solver syntax.
        """ 
                
        for t in self.sets.t:   
            
            # Definitions for fast calcualtion
            Unit_avaiablity = timeSeries[unit][t]
            unit_var = self.U[unit].var

            #Battery related constraints
            if unit == "Battery":  
                m.Add(sum(unit_var.pLine[t][pql] * pqlData[pql]['eMax'] for pql in pqlData)  == unit_var.eNet[t],
                name=f"{unit}.cts.eNet.t{t}")
            
                m.Add(sum(unit_var.pLine[t][pql] for pql in pqlData)  == unit_var.pNet[t],  
                name=f"{unit}.cts.pNet.t{t}") 
                
                m.Add( unit_var.on[t] == sum(unit_var.PQLine[t][pql] for pql in pqlData),
                name=f"{unit}.cts.on.t{t}")  


                m.Add( unit_var.totalCosts[t] == unit_var.pCosts[t] + unit_var.start[t] * Setup['startCosts'], 
                name=f"{unit}.cts.totCosts.t{t}")
                
                
                m.Add( unit_var.pCosts[t] ==  - unit_var.pNet[t] * self.data.timeSeries['Elspot'][t]
                + sum(unit_var.pPOT[t][pql] for pql in pqlData),
                name=f"{unit}.cts.pCosts.t{t}")
                      
            
                #This constraint tries to ties up all NEM capacities:
                m.Add( sum(unit_var.ancService[t][AncS] * self.data.ancService['Up'][AncS] * (1 + self.data.ancService['NEM'][AncS]) for AncS in self.sets.a) <=  loadPoints[('pMax','ub')] - unit_var.pNet[t], 
                name=f"{unit}.cts.ancMaxUp.t{t}")

                m.Add( sum(unit_var.ancService[t][AncS] * self.data.ancService['Dwn'][AncS]  * (1 + self.data.ancService['NEM'][AncS]) for AncS in self.sets.a) <= -loadPoints[('pMin','lb')] + unit_var.pNet[t], 
                name=f"{unit}.cts.ancMaxDown.t{t}")
                    

            #Thermal constraints
            else:  
                m.Add(sum(unit_var.qLine[t][pql] for pql in pqlData) + unit_var.Bypass[t] - unit_var.Condens[t] * pqlData.loc['Cv-line'].max() == unit_var.qNet[t],
                name=f"{unit}.cts.qNet.t{t}")
            
                m.Add(sum(unit_var.pLine[t][pql] for pql in pqlData) - unit_var.Bypass[t] + unit_var.Condens[t]  == unit_var.pNet[t], 
                name=f"{unit}.cts.pNet.t{t}") 
    
                m.Add(sum(unit_var.fLine[t][pql] for pql in pqlData) == unit_var.fNet[t],
                name=f"{unit}.cts.fNet.t{t}") 
                
                m.Add( unit_var.on[t] == sum(unit_var.PQLine[t][pql] for pql in pqlData),
                name=f"{unit}.cts.on.t{t}")  
                
                m.Add( unit_var.pNet[t] * pqlData.loc['direction'].max()  >= unit_var.on[t] * pqlData.loc['pMin',:].min(),
                name=f"{unit}.cts.Bypass.t{t}")       

                m.Add( unit_var.totalCosts[t] == unit_var.pCosts[t]+ unit_var.fCosts[t] + unit_var.qCosts[t] + unit_var.start[t] * Setup['startCosts'], 
                name=f"{unit}.cts.totCosts.t{t}")
                              
                m.Add( unit_var.pCosts[t] ==  - unit_var.pNet[t] * self.data.timeSeries['Elspot'][t] + sum(unit_var.pPOT[t][pql] for pql in pqlData),
                name=f"{unit}.cts.pCosts.t{t}")
                
                m.Add( unit_var.qCosts[t] == sum((unit_var.qLine[t][pql] + unit_var.Bypass[t] - unit_var.Condens[t] * pqlData[pql]['Cv-line']) * pqlData[pql]['qMC'] for pql in pqlData), #unit_var.qNet[t] *  Setup['qMC'],
                name=f"{unit}.cts.qCosts.t{t}") 
    
                m.Add( unit_var.fCosts[t] == sum(unit_var.fLine[t][pql]*pqlData[pql]['fMC'] for pql in pqlData),
                name=f"{unit}.cts.fCosts.t{t}")  
                  

            # General units constraints
            m.Add( unit_var.pNet[t] >=  loadPoints[('pMin','lb')] + max(pqlData.loc['pMin',:].min(),0) * unit_var.on[t] + sum(unit_var.ancService[t][AncS] *self.data.ancService['Dwn'][AncS] for AncS in self.sets.a),
            name=f"{unit}.cts.downReg.t{t}")  
            
            m.Add( unit_var.pNet[t] <=  loadPoints[('pMax','ub')] * Unit_avaiablity - sum(unit_var.ancService[t][AncS] *self.data.ancService['Up'][AncS] for AncS in self.sets.a),
            name=f"{unit}.cts.upReg.t{t}") 

            m.Add( unit_var.ASRevenue[t] ==  sum(unit_var.ancService[t][AncS] * (self.data.timeSeries[AncS+'_Price'][t] - self.data.ancService['LoadFactor'][AncS] * pqlData.loc['Ancillary service cost'].max())  for AncS in self.sets.a),
            name=f"{unit}.cts.ASRevenue.t{t}")             
                                     
            for ancS in self.sets.a:
                 m.Add(unit_var.ancService[t][ancS] * self.data.ancService['Running'][ancS]
                       <= unit_var.on[t]*(loadPoints[('pAbs')] * ancService[ancS]),
                 name=f"{unit}.cts.AncsOn.t{t}.{ancS}")                         
            
            
            # Ensures direct heat transfer to consumption - not storage (PQL1 = low temperature) 
            # if unit == 'VP':
                # self.U[unit].cts.TempLimit[t]      = m.Add(unit_var.qLine[t]['PQL1'] <= self.data.timeSeries['Heat_Demand'][t]*1,
                #                                   name=f"{unit}.cts.TempLimit.t{t}")  




    # -------- Hourly unit constraints from hour 2 and forth --------   
            if t > self.sets.t[0]:
                t_0 = self.sets.t[self.sets.t.index(t)-1]  
                
                m.Add( sum(unit_var.pLine[t][pql] * pqlData[pql]['direction'] for pql in pqlData ) - sum(unit_var.pLine[t_0][pql]  * pqlData[pql]['direction'] for pql in pqlData)
                <=   sum(unit_var.PQLine[t][pql] * (pqlData[pql]['RampUp'] - pqlData.loc['RampUp'].min()) for pql in pqlData) *  Unit_avaiablity +  pqlData.loc['RampUp'].min() ,
                name=f"{unit}.cts.pHourRampUp.t{t}")

                m.Add( -sum(unit_var.pLine[t][pql] * pqlData[pql]['direction'] for pql in pqlData ) + sum(unit_var.pLine[t_0][pql]  * pqlData[pql]['direction'] for pql in pqlData)
                <=    sum(unit_var.PQLine[t][pql] * (pqlData[pql]['RampDown']  -  pqlData.loc['RampDown'].min()) for pql in pqlData) * Unit_avaiablity + pqlData.loc['RampDown'].min() ,
                name=f"{unit}.cts.pHourRampDw.t{t}")        
                
                m.Add(unit_var.start[t] >=  unit_var.on[t] - unit_var.on[t_0],
                name=f"{unit}.cts.start.t{t}")
                

            elif (t == self.sets.t[0]) and (self.data.batch>0):
                
                m.Add( sum(unit_var.pLine[t][pql] * pqlData[pql]['direction'] for pql in pqlData ) - df[unit+'.P'].iloc[-1]
                <=  sum(unit_var.PQLine[t][pql] * (pqlData[pql]['RampUp']  - pqlData.loc['RampUp'].min()) for pql in pqlData)  *   Unit_avaiablity +  pqlData.loc['RampUp'].min(), 
                name=f"{unit}.cts.pHourRampUp.t{t}")

                m.Add( -sum(unit_var.pLine[t][pql] * pqlData[pql]['direction']for pql in pqlData) + df[unit+'.P'].iloc[-1]
                <=   sum(unit_var.PQLine[t][pql] * (pqlData[pql]['RampDown'] - pqlData.loc['RampDown'].min()) for pql in pqlData) *  Unit_avaiablity + pqlData.loc['RampDown'].min(), 
                name=f"{unit}.cts.pHourRampDw.t{t}")      
                
                m.Add(unit_var.start[t] >=  unit_var.on[t] - df[unit+'.On'].iloc[-1],
                name=f"{unit}.cts.start.t{t}")

            
            #Loops the operational lines
            for pql in list(pqlData):   
                pql_lookup = pqlData[pql]                    
                                
                if unit == "Battery":
                    m.Add(unit_var.pPOT[t][pql] == (unit_var.pLine[t][pql] * (-1 if pql_lookup['direction'] == -1 else 1)) * (pql_lookup['pMC'] + (self.data.timeSeries['Tariffs'][t] if pql_lookup['direction'] == -1 else 0)),
                    name=f"{unit}.cts.pPOT.t{t}.{pql}")    
                    
                    m.Add(unit_var.pLine[t][pql] >= unit_var.PQLine[t][pql] * pql_lookup['pMin'],
                    name=f"{unit}.cts.pLineMin.t{t}.{pql}")  
                    
                    m.Add(unit_var.pLine[t][pql] <= unit_var.PQLine[t][pql] * pql_lookup['pMax'],
                    name=f"{unit}.cts.pLineMax.t{t}.{pql}")
                    
                else:
                    
                    # if production of Q, then we fix the power tothe Q variable. If not P is "free". 
                    if pql_lookup['qMin']>0:
                        m.Add(unit_var.pLine[t][pql] == (unit_var.qLine[t][pql] * pql_lookup['a_qp'] + unit_var.PQLine[t][pql] * pql_lookup['b_qp'])/Unit_avaiablity,
                        name=f"{unit}.cts.pLine.t{t}.{pql}")                        

                    m.Add(unit_var.qLine[t][pql] >= unit_var.PQLine[t][pql] * pql_lookup['qMin'],
                    name=f"{unit}.cts.qLineMin.t{t}.{pql}")  
                    
                    m.Add(unit_var.qLine[t][pql] <= unit_var.PQLine[t][pql] * pql_lookup['qMax'],
                    name=f"{unit}.cts.qLineMax.t{t}.{pql}")
                         
                    m.Add(unit_var.fLine[t][pql] == unit_var.qLine[t][pql] * pql_lookup['a_qf'] + unit_var.PQLine[t][pql] * pql_lookup['b_qf'],
                    name=f"{unit}.cts.fLine.t{t}.{pql}")     
    
                    m.Add(unit_var.pPOT[t][pql] == (unit_var.pLine[t][pql] * (-1 if pql_lookup['direction'] == -1 else 1) - unit_var.Bypass[t] + unit_var.Condens[t]) * (pql_lookup['pMC'] + (self.data.timeSeries['Tariffs'][t] if pql_lookup['direction'] == -1 else 0)),
                    name=f"{unit}.cts.pPOT.t{t}.{pql}")   
                    


            # Creates the FCR 4h block
            if "FCR" in self.sets.a:
                hour = t.hour
                if hour % 4 == 0:
                    t_4hblock = t  
                
                elif (hour % 4 != 0) and (self.data.batch>0):
                    m.Add(unit_var.ancService[t]['FCR'] == df[unit+'.FCR'].iloc[-1],
                                                  name=f"{unit}.cts.FCR4h.t{t}") 
                
                else:
                    m.Add(unit_var.ancService[t]['FCR'] == unit_var.ancService[t_4hblock]['FCR'],
                                                  name=f"{unit}.cts.FCR4h.t{t}")  


    
    def set_up_energyBalance(self,m,timeSeries,qStorage,df,batch,end):
        r""" 
        Set all primary purpose constraints and variables up - e.g. Heat/hydrogen or something else
        """

        # Variables
        self.var.Storage       = {}
        self.var.charge        = {}
        self.var.discharge     = {}
             

        for t in self.sets.t: 
            self.var.Storage[t]             = m.NumVar(lb=0, ub=int(qStorage.loc['StorageSize_MWh']), name=f"var.Storage.t{t}")
            self.var.charge[t]              = m.NumVar(lb=0, ub=int(qStorage.loc['Charge_MW']), name=f"var.charge.t{t}")
            self.var.discharge[t]           = m.NumVar(lb=0, ub=int(qStorage.loc['Discharge_MW']), name=f"var.discharge.t{t}")


            m.Add(self.data.timeSeries['Heat_Demand'][t] == sum(self.U[unit].var.qNet[t] for unit in self.sets.plants) + self.var.discharge[t] - self.var.charge[t],  
            name=f"cts.HeatBalance.t{t}")      
           
            # Temp_limt
            # self.cts.tempLimits[t]         = m.Add(self.data.timeSeries['Heat_Demand'][t] * 70 <= sum(self.U[unit].var.qNet[t] *  pqlData[pql]['Temperature'] for pql in pqlData for unit in self.sets.u if unit != "Battery"),  
            #                                 name=f"cts.tempLimits.t{t}")      
            
            #Ensure heat restrictions in Thermal Storage
            m.Add( self.var.Storage[t] <= int(qStorage.loc['StorageSize_MWh'])- sum(sum(self.U[unit].var.ancService[t][anc] for unit in self.sets.plants) * self.data.ancService['Up'][anc] * self.data.ancService['EnergyReservation'][anc] for anc in self.sets.a),
            name=f"cts.storageRegUp.t{t}")

            m.Add(self.var.Storage[t] >= sum(sum(self.U[unit].var.ancService[t][anc] for unit in self.sets.plants) * self.data.ancService['Dwn'][anc] * self.data.ancService['EnergyReservation'][anc] for anc in self.sets.a),
            name=f"cts.storageRegdw.t{t}")
                                                                                          
            
            if (t == self.sets.t[0]) and (self.data.batch == 0):
                m.Add(self.var.Storage[t] == int(qStorage.loc['iniStorage_MWh']) + self.var.charge[t] - self.var.discharge[t],  
                name=f"cts.StorageBal.t{t}")
                
            elif (t == self.sets.t[0]) and (self.data.batch > 0):
                m.Add(self.var.Storage[t] == df.Storage.iloc[-1] + self.var.charge[t] - self.var.discharge[t],  
                name=f"cts.StorageBal.t{t}")
                                   
            else:
                t_0 = self.sets.t[self.sets.t.index(t)-1] # pd.Timedelta(hours=1)   
                
                m.Add(self.var.Storage[t] == self.var.Storage[t_0] + self.var.charge[t] - self.var.discharge[t],  
                name=f"cts.StorageBal.t{t}")           



    def set_up_battery(self,m,timeSeries,BatStorage,df,batch,end):
        r""" 
        Sets all the energy related variables and constraints for the battery storage
        """
        # imports the storage segment information
        socSets = BatStorage.index       

        # Create  Variables
        self.var.SoC = {}
        self.var.socLim = {}
        self.var.socSeg = {}
        self.var.socCost = {}
                    
        for t in self.sets.t: 
            # Lookups for fast creation
            bat_var = self.U['Battery'].var 
            bat_store_var = self.data.Setup['Battery']
            
            #Creates the variables "BatteryStorage
            self.var.SoC[t]             = m.NumVar(lb=0, ub=bat_store_var['Energy Storage'],name=f"var.SoC.t{t}") 
            self.var.socCost[t]         = m.NumVar(lb=0, ub= m.infinity(),name=f"var.socCost.t{t}") 
            self.var.socSeg[t]          = {}
            

            # Initiate the battery start condition at batch 0
            if (t == self.sets.t[0]) and (batch == 0):
                m.Add(self.var.SoC[t] == bat_store_var['Initial Storage'] - bat_var.eNet[t],  
                                              name=f"cts.BatSoCBal.t{t}") 
                
            # Initiate the battery start condition after batch    
            elif (t == self.sets.t[0]) and (batch > 0):
                m.Add(self.var.SoC[t] == df['Battery.SoC'].iloc[-1] - bat_var.eNet[t],  
                                           name=f"cts.BatSoCBal.t{t}")   
                
            # General SoC energybalance
            else: 
                t_0 = self.sets.t[self.sets.t.index(t)-1]
                
                m.Add(self.var.SoC[t] == self.var.SoC[t_0] - bat_var.eNet[t],  
                                             name=f"cts.BatSoCBal.t{t}") 
                
            # Initiate the battery end condition if end of optmization period
            if (t == self.sets.t[-1]) and (batch == end-1):
                m.Add(self.var.SoC[t] >= bat_store_var['Initial Storage'],  
                                              name=f"cts.socEnd.t{t}")     
                
            
            # Ensures energy content to absorb downregulation 
            m.Add(self.var.SoC[t]  <= bat_store_var['Energy Storage'] - sum(bat_var.ancService[t][anc] * self.data.ancService['Dwn'][anc] * self.data.ancService['EnergyReservation'][anc] for anc in self.sets.a) ,
                                                name=f"cts.batStorageRegdw.t{t}")                                  
            
            # Ensures energy content to deliver power for upregulation
            m.Add(self.var.SoC[t]  >= sum(bat_var.ancService[t][anc] * self.data.ancService['Up'][anc] * self.data.ancService['EnergyReservation'][anc] for anc in self.sets.a) ,
                                                name=f"cts.batStorageRegUp.t{t}")   
            
            
            
            
            # Ties the SoC to the segment
            for seg in socSets:
                
                # Creates the segments
                self.var.socSeg[t][seg] = m.IntVar(lb=0, ub=1, name=f"var.socSeg.t{t}.{seg}")  #self.data.Setup['Battery']['Energy Segment2']
                
            #TODO: Check if this is to be rewritting - sum? or just loop the segs?        
            m.Add(self.var.SoC[t] <= sum(self.var.socSeg[t][seg] * BatStorage["upper"][seg]*bat_store_var['Energy Storage'] for seg in socSets), name=f"cts.socReqMax.t{t}.{seg}")
            m.Add(self.var.SoC[t] >= sum(self.var.socSeg[t][seg] * BatStorage["lower"][seg]*bat_store_var['Energy Storage'] for seg in socSets), name=f"cts.socReqMin.t{t}.{seg}")
            
            
            # Hourly costs
            m.Add(self.var.socCost[t] == sum( self.var.socSeg[t][seg] * BatStorage["cost"][seg] for seg in socSets),
            name=f"cts.socCost.t{t}")      

                
    def set_up_powerBalance(self,m,timeSeries,BatStorage,df,batch,end):
        r"""
        Builds the power related constraints and/or variables. This could be a minimum power e.g. sold to market
        or other power related elements
        
        """         

        # Ensures that you cant sell more FFR than tendered         
        if "FFR" in self.sets.a:
            for t in self.sets.t: 
                m.Add(sum(self.U[unit].var.ancService[t]['FFR'] for unit in self.sets.u) <= self.data.timeSeries['FFR_PurchasedMW'][t],
                name=f"{unit}.cts.FFRlimit.t{t}") 
        
        # Creates the limits of sold capacity
        for ancs in self.sets.a:
            if self.data.soldCapacity[ancs].any() > 0:  
                for t in self.sets.t:                    
                    m.Add(sum(self.U[unit].var.ancService[t][ancs] for unit in self.sets.u) == self.data.soldCapacity[ancs][t],
                    name=f"cts.soldCap.t{t}.{ancs}") 
        
        # If DA obligations                     
        if self.data.soldCapacity["DA"].any() > 0:  
            for t in self.sets.t: 
                m.Add(sum(self.U[unit].var.pNet[t] for unit in self.sets.u) == self.data.soldCapacity["DA"][t],
                 name=f"cts.soldDA.t{t}") 
                    
        
            
    def set_up_objective_function(self,m):
        r"""
        Builds the object function of the model - this is a minimization function, where negative means profits
        """

        self.objective = m.Minimize(
            sum(self.U[u].var.totalCosts[t] - self.U[u].var.ASRevenue[t] 
            + (self.var.socCost[t] if "Battery" in self.sets.u else 0) for t, u in itertools.product(self.sets.t,self.sets.u))/1e6)
            
                                            
                                                        
    def extract_results_from_solver(self):
        r"""
        Eksporterer modellens optimale løsning.
        """

        data_dict = {}

        
        for t in self.sets.t:
            
            data_dict[t] = {
                'Elspot': self.data.timeSeries['Elspot'][t],
                'Storage': self.var.Storage[t].solution_value(),
                'Charge': self.var.charge[t].solution_value(),
                'Discharge': self.var.discharge[t].solution_value(),
                'Heat Demand': self.data.timeSeries['Heat_Demand'][t],
            }
            
    
            for u in self.sets.u:
                data_dict[t][u + '.On'] = self.U[u].var.on[t].solution_value()
                data_dict[t][u + '.Starts'] = self.U[u].var.start[t].solution_value()
                data_dict[t][u + '.P'] = self.U[u].var.pNet[t].solution_value()
                
                if u == "Battery":
                    data_dict[t][u + '.SoC'] = self.var.SoC[t].solution_value()
                    data_dict[t][u + '.P_storage'] = self.U[u].var.eNet[t].solution_value()
                else:
                    data_dict[t][u + '.Q'] = self.U[u].var.qNet[t].solution_value()
                    data_dict[t][u + '.bypass'] = self.U[u].var.Bypass[t].solution_value()
                    data_dict[t][u + '.Condens'] = self.U[u].var.Condens[t].solution_value()              
                    data_dict[t][u + '.F'] = self.U[u].var.fNet[t].solution_value()
                
                for AncS in self.sets.a:
                    data_dict[t][u + '.' + AncS] = self.U[u].var.ancService[t][AncS].solution_value()
                    
                data_dict[t][u + '.TotalCosts'] = self.U[u].var.totalCosts[t].solution_value()
                data_dict[t][u + '.pCosts'] = self.U[u].var.pCosts[t].solution_value()
                data_dict[t][u + '.AS_Revenue'] = self.U[u].var.ASRevenue[t].solution_value()
                
                activePQLine = [self.U[u].var.PQLine[t][pql].solution_value() for pql in self.data.pqlData[u]]
                active_line_segment = [pql for pql, pqlOnOff in zip(self.data.pqlData[u], activePQLine) if pqlOnOff == 1]
                data_dict[t][u + '.ActiveLine'] = (active_line_segment[0] if active_line_segment else None)
                
                for pql in self.data.pqlData[u]:
                    data_dict[t][u + '.' + pql] = self.U[u].var.PQLine[t][pql].solution_value()
                

        data_dict[self.sets.t[0]]['TotalCost'] = self.model.Objective().Value()
    
        data = pd.DataFrame.from_dict(data_dict, orient='index')
            
        
        return data  
