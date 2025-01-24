# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:11:37 2025

@author: JWB
"""

from ortools.math_opt.python import mathopt
import pandas as pd
import itertools
from Helper_functions import *
from datetime import timedelta

def run_plan_optimization(
                 loop_timeseries,
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
     
    optimizer = plan_optimizer(tSet, uSet, aSet, loop_timeseries, qStorage, BatStorage, Setup, pqlData, ancServices, df, num, end, loop_soldCap,maxTime)
    
    optimizer.run()
    
    df = optimizer.extract_results_from_solver()

    return df



class plan_optimizer:
    """
    Optimization class for multi-timestep models using OR-Tools MathOpt. 
    """

    def __init__(self, tSet, uSet, aSet, loop_timeseries, qStorage, BatStorage, Setup, pqlData, ancServices, df, num, end, loop_soldCap,maxTime):
        """
        Initialize the optimizer with the given number of timesteps.
        """
        
        #Creates empty dictionaries - fast approach for large datasets
        self.sets = {}
        self.data = {}
        self.var = {}

         
        # Structuring the input data
        self.load_data(tSet, uSet, aSet, loop_timeseries, qStorage, BatStorage, Setup, pqlData, ancServices, df, num, end, loop_soldCap)

        
        #Model params
 
        self.params = mathopt.SolveParameters(enable_output=False, time_limit=timedelta(seconds=maxTime*60),relative_gap_tolerance = 0.01) 
        self.model = mathopt.Model(name="dispatcher")
        
        
        # Set up model variables and constraints
        self.set_up_model_variables_and_constraints()

    def run(self):
        """
        Solve the optimization model.
        """
        
        self.result = mathopt.solve(self.model, mathopt.SolverType.HIGHS, params= self.params) #GSCIP;HIGHS;CP_SAT
        
        if self.result.termination.reason != mathopt.TerminationReason.OPTIMAL:
            raise RuntimeError(f"Model failed to solve: {self.result.termination}")

        print('Problem solved in',self.result.solve_time()) # To call outside use: optimizer.result.solve_time()            
        print('Objective value, MDKK =', self.result.objective_value()) # To call outside use: optimizer.result.objective_value()
        print('gap [%] =', (round(1 - self.result.objective_value() / self.result.dual_bound(), 4) * 100))

    def load_data(self,tSet,uSet,aSet,loop_timeseries,qStorage, BatStorage, Setup ,pqlData,ancServices,df,num,end,loop_soldCap):
        r"""
        Reads and interprets input data
        """
        
        
        #Creates the sets  
        self.sets = {
            't': tSet,
            'units': uSet,
            'anc': aSet,
            'plants': Setup.columns[Setup.loc['asset type'] == 'plant'].tolist()       
            }
           
        
        
        #Creates the datasets
        self.data = {
            'timeSeries': loop_timeseries,
            'qStorage': qStorage,
            'BatStorage': BatStorage,
            'Setup': Setup,
            'pqlData': pqlData,
            'ancService': ancServices,
            'df': df,
            'batch': num,
            'batchEnd': end,
            'soldCapacity': loop_soldCap
            }
        
        

    def set_up_model_variables_and_constraints(self):
        """
        Define variables, constraints, and objectives for multiple timesteps for the optimization problem.
        """
        
        # Loops over each unit and calls a function to set up unit variables and constraints
        for unit in self.sets['units']:
            print('creating unit variables & constraints:',unit)
            self.set_up_unit_variables_and_constraints(unit, self.data['Setup'][unit],self.data['timeSeries'], self.data['pqlData'][unit], self.data['ancService'][unit], self.data['df'])
            
            # If units contain a battery futher requirements are needed
            if unit == "Battery":
                self.set_up_battery(self.data['timeSeries'], self.data['BatStorage'], self.data['df'], self.data['batch'], self.data['batchEnd'])
                
        # General constraints and objective
        self.set_up_energyBalance(self.data['timeSeries'], self.data['qStorage'], self.data['df'], self.data['batch'], self.data['batchEnd'])
        self.set_up_powerBalance(self.data['timeSeries'], self.data['df'], self.data['batch'], self.data['batchEnd'])
        self.set_up_objective_function()


    def set_up_unit_variables_and_constraints(self, unit,Setup,timeSeries,pqlData,ancService,df):

        """
        Define variables and constraints specific to each unit across timesteps.
        """
        
        # Create dictionaries to store variables for each timestep for the current unit
        unit_vars = {}
        anc_service_vars = {}        
        pql_vars = {}
        
        
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

             
        
        for t in self.sets['t']:  # Loop over timesteps
        
            # Precalcualted input for faster calculation
            Unit_avaiablity = timeSeries[unit][t]
        
            ## -------- Builds variables  ------------------------------------------------------------

            # Create variables for each timestep for the current unit
            pNet = self.model.add_variable(lb = loadPoints[('pMin','lb')] * Unit_avaiablity, ub = loadPoints[('pMax','ub')] * Unit_avaiablity, name=f"pNet_{unit}_{t}") 
            totalCosts  =  self.model.add_variable(lb=float('-inf'), ub=float('inf'), name=f"totalCosts_{unit}_{t}")
            pCosts =  self.model.add_variable(lb=float('-inf'), ub=float('inf'), name=f"pCosts_{unit}_{t}")
            ancRevenue  =  self.model.add_variable(lb=float('-inf'), ub=float('inf'), name=f"ancRevenue_{unit}_{t}") 
            on  =  self.model.add_binary_variable(name=f"on_{unit}_{t}")
            start =  self.model.add_binary_variable(name=f"start_{unit}_{t}")
            
            if unit == "Battery":
                eNet   =  self.model.add_variable(lb=loadPoints[('pMin','lb')] * Unit_avaiablity, ub=loadPoints[('eMax','ub')] * Unit_avaiablity, name=f"eNet_{unit}_{t}")
            else:
                fNet =  self.model.add_variable(lb=0, ub=loadPoints[('fMax','ub')] * Unit_avaiablity, name=f"fNet_{unit}_{t}")  
                qNet = self.model.add_variable(lb=0, ub= loadPoints[('qMax','ub')] * Unit_avaiablity, name=f"qNet_{unit}_{t}")
                Condens =  self.model.add_variable(lb=0, ub= loadPoints[('qMax','ub')] * pqlData.loc['Cv-line'].max(), name=f"Condens_{unit}_{t}")  
                Bypass = self.model.add_variable(lb=0, ub= loadPoints[('pMax','ub')] * pqlData.loc['Bypass'].max(), name=f"Bypass_{unit}_{t}")
                qCosts =  self.model.add_variable(lb=0, ub=float('inf'), name=f"qCosts_{unit}_{t}")
                fCosts =  self.model.add_variable(lb=float('-inf'), ub=float('inf'), name=f"fCosts_{unit}_{t}")
                temp =  self.model.add_variable(lb=0, ub=float('inf'), name=f"temp_{unit}_{t}")


        ## -------- Builds "nested" variables (addional loop) ------------------------------------------------------------
            
            # Loop over ancillary services to create variabel
            anc_service_vars[t] = {}       
            for anc in self.sets['anc']:  
                anc_service_vars[t][anc] = self.model.add_variable(lb=0, ub=loadPoints[('pAbs')] * ancService[anc] * Unit_avaiablity,
                name=f"ancService_{unit}_{anc}_{t}")
               
                
            # Loop over each pql, line segments that defines operation
            pql_vars[t] = {}
            for pql in pqlData:
                pql_vars[t][pql] = {
                    "PQLine": self.model.add_binary_variable(name=f"PQLine_{unit}_{t}_{pql}"),
                    "pLine": self.model.add_variable(lb=min(0, min(pqlData[pql]['pMax'], pqlData[pql]['pMin'])), ub=max(0, pqlData[pql]['pMax']), name=f"pLine_{unit}_{t}_{pql}"),
                    "pPOT": self.model.add_variable(lb=float('-inf'), ub=float('inf'),name=f"pPOT_{unit}_{t}_{pql}"),
                    "qLine": self.model.add_variable( lb=0, ub=pqlData[pql]['qMax'],name=f"qLine_{unit}_{t}_{pql}") if unit != "Battery" else None,
                    "fLine": self.model.add_variable( lb=0, ub=pqlData[pql]['fMax'],name=f"fLine_{unit}_{t}_{pql}") if unit != "Battery" else None,
                }
            
 
            # Stores all the unit varibales for the timestamp
            unit_vars[t] = {
              "pNet": pNet,
              "totalCosts": totalCosts,
              "pCosts" :pCosts,
              "ancRevenue": ancRevenue,
              "ancServices": anc_service_vars[t],
              "pql":pql_vars[t],
              "on": on,
              "start": start,
              "qNet": qNet if unit != "Battery" else None,
              "fNet": fNet if unit != "Battery" else None,
              "qCosts": qCosts if unit != "Battery" else None,
              "fCosts": fCosts if unit != "Battery" else None,             
              "temp": temp if unit != "Battery" else None, 
              "Bypass": Bypass if unit != "Battery" else None,
              "Condens": Condens if unit != "Battery" else None,
              "eNet": eNet if unit == "Battery" else None,
              }
    
        # Stores unit variables in a main var dictionary
        self.var[unit] = unit_vars
        
        #Calls the set_up_unit_constraints to create constraints
        self.set_up_unit_constraints(unit,Setup,timeSeries,pqlData,ancService,df,loadPoints)
        
        
    def set_up_unit_constraints(self, unit,Setup,timeSeries,pqlData,ancService,df,loadPoints):        
        """
        Function called from set_up_unit_variables_and_constraints to create the constraints.
        """ 
           
        ## -------- Builds Unit Constraints ------------------------------------------------------------        
        # Create and add constraints directly to the model without saving in main dict
        for t in self.sets['t']:
            
            # Precalcualted input for faster calculation
            Unit_avaiablity = timeSeries[unit][t]
            unit_vars = self.var[unit][t]
            unit_vars_pql = self.var[unit][t]['pql']
            
            ## Creates battery specific constraints
            if unit == "Battery":
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["pLine"] * pqlData[pql]['eMax'] for pql in pqlData) == unit_vars["eNet"],
                    name=f"{unit}_cts_eNet_{t}")           
                
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["pLine"] for pql in pqlData) == unit_vars["pNet"],
                    name=f"{unit}_cts_pNet_{t}")
  
                self.model.add_linear_constraint(
                    unit_vars['totalCosts']  == unit_vars['pCosts'] + unit_vars['start'] * Setup['startCosts'], 
                    name=f"{unit}_cts_totCosts_{t}")
                               
                self.model.add_linear_constraint(
                    unit_vars['pCosts'] ==  - unit_vars["pNet"] * timeSeries['Elspot'][t] + sum(unit_vars_pql[pql]["pPOT"] for pql in pqlData),
                    name=f"{unit}_cts_pCosts_{t}")                

                #This constraint tries to ties up all NEM capacities:
                self.model.add_linear_constraint(
                    sum(unit_vars["ancServices"][AncS] * self.data['ancService']['Up'][AncS] * (1 + self.data['ancService']['NEM'][AncS]) for AncS in self.sets['anc']) <=  loadPoints[('pMax','ub')] - unit_vars["pNet"], 
                    name=f"{unit}_cts_ancMaxUp_{t}")

                self.model.add_linear_constraint(
                    sum(unit_vars["ancServices"][AncS] * self.data['ancService']['Dwn'][AncS]  * (1 + self.data['ancService']['NEM'][AncS]) for AncS in self.sets['anc']) <= -loadPoints[('pMin','lb')] + unit_vars["pNet"], 
                    name=f"{unit}_cts_ancMaxDown_{t}")
                    
                
            
            ## Creates thermal plant specific constraints
            else:
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["qLine"] for pql in pqlData) + unit_vars["Bypass"] - unit_vars["Condens"] * pqlData.loc['Cv-line'].max() == unit_vars["qNet"],
                    name=f"{unit}_cts_qNet_{t}")
                
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["pLine"] for pql in pqlData) - unit_vars["Bypass"] + unit_vars["Condens"] == unit_vars["pNet"],
                    name=f"{unit}_cts_pNet_{t}")
                
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["fLine"] for pql in pqlData) == unit_vars["fNet"],
                    name=f"{unit}_cts_fNet_{t}")
                
                self.model.add_linear_constraint(
                    sum(pqlData[pql]["Temperature"] * (unit_vars_pql[pql]["qLine"] + unit_vars["Bypass"] - unit_vars["Condens"] * pqlData.loc['Cv-line'].max()) for pql in pqlData) == unit_vars["temp"],
                    name=f"{unit}_cts_temp_{t}")
                
                self.model.add_linear_constraint(
                    unit_vars["pNet"] * pqlData.loc['direction'].max() >= unit_vars["on"] * pqlData.loc['pMin',:].min(),
                    name=f"{unit}_cts_Bypass_{t}")

                self.model.add_linear_constraint(
                    unit_vars['totalCosts']  == unit_vars['pCosts'] + unit_vars['fCosts'] + unit_vars['qCosts'] + unit_vars['start'] * Setup['startCosts'], 
                    name=f"{unit}_cts_totCosts_{t}")              
                
                self.model.add_linear_constraint(
                    unit_vars['pCosts'] ==  - unit_vars["pNet"] * timeSeries['Elspot'][t]+ sum(unit_vars_pql[pql]["pPOT"] for pql in pqlData),
                    name=f"{unit}_cts_pCosts_{t}")
                
                self.model.add_linear_constraint(
                    unit_vars['qCosts'] == sum((unit_vars_pql[pql]["qLine"] + unit_vars["Bypass"] - unit_vars["Condens"] * pqlData[pql]['Cv-line']) * pqlData[pql]['qMC'] for pql in pqlData), 
                    name=f"{unit}_cts_qCosts_{t}") 
    
                self.model.add_linear_constraint(
                    unit_vars['fCosts'] == sum(unit_vars_pql[pql]["fLine"] * pqlData[pql]['fMC'] for pql in pqlData),
                    name=f"{unit}_cts_fCosts_{t}")  



            ## Generic Constraints for all unit types          
            self.model.add_linear_constraint(
                unit_vars["on"] == sum(unit_vars_pql[pql]['PQLine'] for pql in pqlData),
                name=f"{unit}_cts_on_{t}")

            self.model.add_linear_constraint(
                unit_vars["pNet"] >=  loadPoints[('pMin','lb')] + max(pqlData.loc['pMin',:].min(),0) * unit_vars["on"] + sum(unit_vars["ancServices"][anc] * self.data['ancService']['Dwn'][anc] for anc in self.sets['anc']),
                name=f"{unit}_cts_downReg_{t}")  
            
            #Appararently what works if no ancillary services are provided. 
            if self.sets['anc']:
                self.model.add_linear_constraint(
                unit_vars["pNet"]  <= loadPoints[('pMax','ub')] * Unit_avaiablity - sum(unit_vars["ancServices"][anc] *self.data['ancService']['Up'][anc] for anc in self.sets['anc']),
                name=f"{unit}_cts_upReg_{t}")

            self.model.add_linear_constraint(
                unit_vars["ancRevenue"] ==  sum(unit_vars["ancServices"][anc]  * (timeSeries[anc+'_Price'][t] - self.data['ancService']['LoadFactor'][anc] * pqlData.loc['Ancillary service cost'].max())  for anc in self.sets['anc']),
                name=f"{unit}_cts_ASRevenue_{t}")        

            for anc in self.sets['anc']:
                  self.model.add_linear_constraint(
                      unit_vars["ancServices"][anc] * self.data['ancService']['Running'][anc] <=  unit_vars["on"] * (loadPoints[('pAbs')] * ancService[anc]),
                      name=f"{unit}_cts_AncsOn_{t}.{anc}")        



             ## Time dependent constraints
            if t > self.sets['t'][0]:        
                
                #prior timestep
                t_0 = self.sets['t'][self.sets['t'].index(t) - 1]
                
                self.model.add_linear_constraint(
                      sum(unit_vars_pql[pql]["pLine"] * pqlData[pql]['direction'] for pql in pqlData) - sum(self.var[unit][t_0]['pql'][pql]["pLine"]  * pqlData[pql]['direction'] for pql in pqlData)
                      <=   sum(unit_vars_pql[pql]['PQLine'] * (pqlData[pql]['RampUp'] - pqlData.loc['RampUp'].min()) for pql in pqlData) *  Unit_avaiablity +  pqlData.loc['RampUp'].min(),
                      name=f"{unit}_cts_pHourRampUp_{t}")
    
                self.model.add_linear_constraint(
                    sum(-unit_vars_pql[pql]["pLine"] * pqlData[pql]['direction'] for pql in pqlData) + sum(self.var[unit][t_0]['pql'][pql]["pLine"]  * pqlData[pql]['direction'] for pql in pqlData)
                    <= sum(unit_vars_pql[pql]['PQLine'] * (pqlData[pql]['RampDown']  -  pqlData.loc['RampDown'].min()) for pql in pqlData) * Unit_avaiablity + pqlData.loc['RampDown'].min(),
                      name=f"{unit}_cts_pHourRampDw_{t}")      
            
                 
                self.model.add_linear_constraint(
                    unit_vars["start"]  >=  unit_vars["on"]  - self.var[unit][t_0]["on"] ,
                    name=f"{unit}_cts_start_{t}")
                

            elif (t == self.sets['t'][0]) and (self.data['batch'] >0):
                
                self.model.add_linear_constraint(
                    sum(unit_vars_pql[pql]["pLine"] * pqlData[pql]['direction'] for pql in pqlData ) - df[unit+'.P'].iloc[-1]
                    <=  sum(unit_vars_pql[pql]['PQLine'] * (pqlData[pql]['RampUp']  - pqlData.loc['RampUp'].min()) for pql in pqlData)  *   Unit_avaiablity +  pqlData.loc['RampUp'].min(), 
                    name=f"{unit}_cts_pHourRampUp_{t}")

                self.model.add_linear_constraint(
                    - sum(unit_vars_pql[pql]["pLine"] * pqlData[pql]['direction']for pql in pqlData) + df[unit+'.P'].iloc[-1]
                    <=   sum(unit_vars_pql[pql]['PQLine'] * (pqlData[pql]['RampDown'] - pqlData.loc['RampDown'].min()) for pql in pqlData) *  Unit_avaiablity + pqlData.loc['RampDown'].min(), 
                    name=f"{unit}_cts_pHourRampDw_{t}")      
                
                self.model.add_linear_constraint(
                    unit_vars["start"] >=  unit_vars["on"]  - df[unit+'.On'].iloc[-1],
                    name=f"{unit}_cts_start_{t}")
            
            
            
            #extracts the hour (format dd-mm-yyyy hh:mm)
            if "FCR" in self.sets['anc']:
                hour = t.hour
                if hour % 4 == 0:
                    t_4hblock = t  
                
                elif (hour % 4 != 0) and (self.data['batch']>0):
                    self.model.add_linear_constraint( unit_vars["ancServices"]['FCR']  ==  df[unit+'.FCR'].iloc[-1],
                    name=f"{unit}.cts.FCR4h.t{t}") 
                
                else:
                    self.model.add_linear_constraint(unit_vars["ancServices"]['FCR']  == self.var[unit][t_4hblock]["ancServices"]['FCR'],
                    name=f"{unit}.cts.FCR4h.t{t}")  
            
            
            
            ## Loops the operational lines
            for pql in list(pqlData):
                pql_lookup = pqlData[pql]
                direction = (-1 if pql_lookup['direction'] == -1 else 1)
                tariff_consumption = (timeSeries['Tariffs'][t] if pql_lookup['direction'] == -1 else 0)                       
            
            
                # Battery specific pql constraints
                if unit == "Battery":
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["pPOT"] == unit_vars_pql[pql]["pLine"] * direction * (pql_lookup['pMC'] + tariff_consumption),
                        name=f"{unit}_cts_pPOT_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["pLine"] >= unit_vars_pql[pql]["PQLine"] * pql_lookup['pMin'],
                        name=f"{unit}_cts_pLineMin_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["pLine"] <= unit_vars_pql[pql]["PQLine"] * pql_lookup['pMax'],
                        name=f"{unit}_cts_pLineMax_{t}_{pql}")
                    
                                              
    
                # Thermal plant specific pql constraints
                else:
                    if pql_lookup['qMin'] > 0:
                        self.model.add_linear_constraint(
                            unit_vars_pql[pql]["pLine"] == (unit_vars_pql[pql]["qLine"] * pql_lookup['a_qp'] + unit_vars_pql[pql]["PQLine"] * pql_lookup['b_qp'])/Unit_avaiablity,
                            name=f"{unit}_cts_pLine_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["qLine"] >= unit_vars_pql[pql]["PQLine"] * pql_lookup['qMin'],
                        name=f"{unit}_cts_qLineMin_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["qLine"] <= unit_vars_pql[pql]["PQLine"] * pql_lookup['qMax'],
                        name=f"{unit}_cts_qLineMax_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["fLine"] == (unit_vars_pql[pql]["qLine"] * pql_lookup['a_qf'] + unit_vars_pql[pql]["PQLine"] * pql_lookup['b_qf']), 
                        name=f"{unit}_cts_fLine_{t}_{pql}")
                    
                    self.model.add_linear_constraint(
                        unit_vars_pql[pql]["pPOT"] == (unit_vars_pql[pql]["pLine"] * direction - unit_vars["Bypass"] + unit_vars["Condens"]) 
                        * (pqlData[pql]['pMC'] + tariff_consumption),
                        name=f"{unit}_cts_pPOT_{t}_{pql}")
                    
                    

    def set_up_energyBalance(self,timeSeries,qStorage,df,batch,end):
        r"""
        Builds the energy balance and the related constraints and energy equations
        """
        
        # Initialize variables for each timestep
        Storage_vars = {}
        
        
        for t in self.sets['t']:  # Loop over timesteps
              
        # Create variables for each timestep for the current unit
            storage = self.model.add_variable(lb=0, ub=int(qStorage.loc['StorageSize_MWh']), name=f"storage_{t}") 
            charge  =  self.model.add_variable(lb=0, ub=int(qStorage.loc['Charge_MW']), name=f"charge_{t}")
            discharge  =  self.model.add_variable(lb=0, ub=int(qStorage.loc['Discharge_MW']), name=f"discharge_{t}")
            
            # Stores all the data
            Storage_vars[t] = {
             "storage": storage,
             "charge": charge,
             "discharge" :discharge
             }
    
        # Stores storage variables in the main var dictionary
        self.var['storage_vars'] = Storage_vars


        for t in self.sets['t']:  # Loop over timesteps    
        
            self.model.add_linear_constraint(
                float(self.data['timeSeries']['Heat_Demand'][t]) == sum(self.var[unit][t]["qNet"] for unit in self.sets['plants']) + self.var['storage_vars'][t]["discharge"] -  self.var['storage_vars'][t]["charge"],  
                name=f"cts_HeatBalance_{t}")      
            
            ## Temperature restriktions
            self.model.add_linear_constraint(
                float(self.data['timeSeries']['Heat_Demand'][t])* 70 <= sum(self.var[unit][t]["temp"] for unit in self.sets['plants']) + 80 *self.var['storage_vars'][t]["discharge"] - 90 *self.var['storage_vars'][t]["charge"],  
                name=f"cts_temperature_{t}")    

            ## Ensures room in thermal storage to do the regulations
            self.model.add_linear_constraint(
                self.var['storage_vars'][t]["storage"] <= int(qStorage.loc['StorageSize_MWh']) 
                - sum(sum(self.var[unit][t]["ancServices"][anc] for unit in self.sets['plants']) * self.data['ancService']['Up'][anc] * self.data['ancService']['EnergyReservation'][anc] for anc in self.sets['anc']),
                name=f"storageRegUp_t{t}")

            self.model.add_linear_constraint(
                self.var['storage_vars'][t]["storage"] >= sum(sum(self.var[unit][t]["ancServices"][anc] for unit in self.sets['plants']) * self.data['ancService']['Dwn'][anc] * self.data['ancService']['EnergyReservation'][anc] for anc in self.sets['anc']),
                name=f"storageRegdw_t{t}")
                                                    

            if (t == self.sets['t'][0]) and (self.data['batch'] == 0):
            
                self.model.add_linear_constraint(
                    self.var['storage_vars'][t]["storage"] == int(qStorage.loc['iniStorage_MWh']) - self.var['storage_vars'][t]["discharge"] +  self.var['storage_vars'][t]["charge"],  
                    name=f"StorageBal_{t}")
                
            elif (t == self.sets['t'][0]) and (self.data['batch'] > 0):
             
                self.model.add_linear_constraint(
                    self.var['storage_vars'][t]["storage"] == df.Storage.iloc[-1] - self.var['storage_vars'][t]["discharge"] +  self.var['storage_vars'][t]["charge"],  
                    name=f"StorageBal_{t}")
                                   
            else:
                t_0 = self.sets['t'][self.sets['t'].index(t) - 1]                
                self.model.add_linear_constraint(
                    self.var['storage_vars'][t]["storage"] == self.var['storage_vars'][t_0]["storage"] - self.var['storage_vars'][t]["discharge"] +  self.var['storage_vars'][t]["charge"],  
                    name=f"StorageBal_{t}")  
           

    def set_up_battery(self,timeSeries,BatStorage,df,batch,end):
        r"""
        Builds the energy balance constraints and equations
        """
        
        # imports the storage segment information
        socSets = BatStorage.index 

        # Initialize variables for each timestep
        battery_vars = {}
        soc_vars = {}
        
        for t in self.sets['t']:  # Loop over timesteps
         
        # Create variables for each timestep for the current unit
            soc = self.model.add_variable(lb=0, ub=self.data['Setup']['Battery']['Energy Storage'], name=f"soc_{t}") 
            socCost  =  self.model.add_variable(lb=0, ub= float('inf'), name=f"socCost_{t}")
            
            soc_segment_vars = {}  
            for seg in socSets:                
                # Create binary variable to indicate whether the soc segment is active
                soc_segment_vars[seg] = self.model.add_binary_variable(name=f"soc_{seg}_{t}")
            
            # Stores all the data
            battery_vars[t] = {
             "soc": soc,
             "socCost": socCost,
             "socSeg" : soc_segment_vars
             }
    
        # Stores storage variables in the main var dictionary
        self.var['battery_vars'] = battery_vars

        
        ## -------- Builds battery related constraints --------------------------------------------------------------------
        for t in self.sets['t']:  # Loop over timesteps
            bat_vars = self.var['Battery'][t]
            bat_storage_var = self.data['Setup']['Battery']
        
    
            # Initiate the battery start condition at batch 0
            if (t == self.sets['t'][0]) and (batch == 0):
                self.model.add_linear_constraint(
                    self.var['battery_vars'][t]['soc'] == bat_storage_var['Initial Storage'] - bat_vars["eNet"],  
                    name=f"cts_BatSoCBal_{t}") 
                
            # Initiate the battery start condition after batch    
            elif (t == self.sets['t'][0]) and (batch > 0):
                self.model.add_linear_constraint(
                    self.var['battery_vars'][t]['soc']  == df['Battery.SoC'].iloc[-1] - bat_vars["eNet"],  
                    name=f"cts_BatSoCBal_{t}")   
                
            # General SoC energybalance
            else: 
                t_0 = self.sets['t'][self.sets['t'].index(t)-1]
                self.model.add_linear_constraint(
                    self.var['battery_vars'][t]['soc'] == self.var['battery_vars'][t_0]['soc'] - bat_vars["eNet"],  
                    name=f"cts_BatSoCBal_{t}") 
                
            # Initiate the battery end condition if end of optmization period
            if (t == self.sets['t'][-1]) and (batch == end-1):
                self.model.add_linear_constraint(
                    self.var['battery_vars'][t]['soc'] >= bat_storage_var['Initial Storage'],  
                    name=f"cts_socEnd_{t}")     
                
            
            # Ensures energy content to absorb downregulation 
            self.model.add_linear_constraint(
                self.var['battery_vars'][t]['soc'] <= bat_storage_var['Energy Storage'] - sum(bat_vars["ancServices"][anc] * self.data['ancService']['Dwn'][anc] * self.data['ancService']['EnergyReservation'][anc] for anc in self.sets['anc']),
                name=f"cts_batStorageRegdw.t{t}")                                  
            
            # Ensures energy content to deliver power for upregulation
            self.model.add_linear_constraint(
                self.var['battery_vars'][t]['soc']  >= sum(bat_vars["ancServices"][anc] * self.data['ancService']['Up'][anc] * self.data['ancService']['EnergyReservation'][anc] for anc in self.sets['anc']),
                name=f"cts_batStorageRegUp.t{t}")   
                      
             
            #TODO: Check if this is to be rewritting - sum? or just loop the segs?        
            self.model.add_linear_constraint(
                self.var['battery_vars'][t]['soc'] <= sum(self.var['battery_vars'][t]['socSeg'][seg] * BatStorage["upper"][seg] * bat_storage_var['Energy Storage'] for seg in socSets), name=f"cts_socReqMax_{t}_{seg}")
            
            self.model.add_linear_constraint(
                self.var['battery_vars'][t]['soc'] >= sum(self.var['battery_vars'][t]['socSeg'][seg] * BatStorage["lower"][seg] * bat_storage_var['Energy Storage'] for seg in socSets), name=f"cts_socReqMin_{t}_{seg}")
                                                     
            
            # Hourly costs
            self.model.add_linear_constraint(
                self.var['battery_vars'][t]['socCost'] == sum( self.var['battery_vars'][t]['socSeg'][seg]  * BatStorage["cost"][seg] for seg in socSets),
                name=f"cts_socCost_{t}")                       
    
            

    def set_up_powerBalance(self,timeSeries,df,batch,end):
        r"""
        Builds power/ancillary service constraints
        """
               
        if "FFR" in self.sets['anc']:
            for t in self.sets['t']: 
                self.model.add_linear_constraint(
                    sum(self.var[unit][t]["ancServices"]['FFR'] for unit in self.sets['units']) <= self.data['timeSeries']['FFR_PurchasedMW'][t],
                    name=f"cts_FFRlimit_{t}") 
                        

        # Creates the limits of sold capacity
        for ancs in self.sets['anc']:
            if self.data['soldCapacity'][ancs].any() > 0:  
                for t in self.sets['t']:
                    self.model.add_linear_constraint(
                        sum(self.var[unit][t]["ancServices"][anc] for unit in self.sets['units']) == self.data['soldCapacity'][ancs][t],
                        name=f"cts_soldCap_{ancs}_{t}") 
                    
        if self.data['soldCapacity']["DA"].any() > 0:  
                for t in self.sets['t']:
                    self.model.add_linear_constraint(
                        sum(self.var[unit][t]["pNet"] for unit in self.sets['units']) == self.data.soldCapacity["DA"][t],
                        name=f"cts_soldDA.t{t}") 

            
    def set_up_objective_function(self):
        r"""
        Bygger modellens objektfunktion.
        """
        print('Optimizing...')
        
        
        self.model.minimize((sum(self.var[unit][t]['totalCosts'] - self.var[unit][t]['ancRevenue']
                            + (self.var['battery_vars'][t]['socCost'] if "Battery" in self.sets['units'] else 0) for t, unit in itertools.product(self.sets['t'], self.sets['units'])))/1e6)
                                       

          
        
    def extract_results_from_solver(self):
        r"""
        Exports the model's optimal solution.
        """
        

        # Convert variable_values to a dictionary for faster lookup
        result_values = {var.name: value for var, value in self.result.variable_values().items()}
        
        
        # Initialize the data dictionary
        data_dict = {}
        
        # Loop through the available timestamps
        for t in self.sets['t']:

            # Format the timestamp string
            timestamp_str = str(t)
            
            # Extract relevant variables efficiently
            data_dict[t] = {
                'Elspot': self.data['timeSeries']['Elspot'][t],
                'Heat Demand': self.data['timeSeries']['Heat_Demand'][t],
                'Storage': result_values.get(f'storage_{timestamp_str}', None),
                'Charge': result_values.get(f'charge_{timestamp_str}', None),
                'Discharge': result_values.get(f'discharge_{timestamp_str}', None),
                }
                
          # Loop through the units and add unit-specific data
            for u in self.sets['units']:
                data_dict[t][f'{u}.On'] = result_values.get(f'on_{u}_{timestamp_str}', None)
                data_dict[t][f'{u}.Starts'] = result_values.get(f'start_{u}_{timestamp_str}', None)
                data_dict[t][f'{u}.P'] = result_values.get(f'pNet_{u}_{timestamp_str}', None)
                
                if u == "Battery":
                    data_dict[t][u + '.SoC'] = result_values.get(f'soc_{timestamp_str}', None)
                    data_dict[t][u + '.P_storage'] = result_values.get(f'eNet_{u}_{timestamp_str}', None)
                else:
                    data_dict[t][u + '.Q'] = result_values.get(f'qNet_{u}_{timestamp_str}', None)
                    data_dict[t][u + '.bypass'] = result_values.get(f'Bypass_{u}_{timestamp_str}', None)
                    data_dict[t][u + '.Condens'] =  result_values.get(f'Condens_{u}_{timestamp_str}', None)    
                    data_dict[t][u + '.F'] =  result_values.get(f'fNet_{u}_{timestamp_str}', None)
                    data_dict[t][u + '.temp'] =  result_values.get(f'temp{u}_{timestamp_str}', None)
                for AncS in self.sets['anc']:
                    data_dict[t][u + '.' + AncS] = result_values.get(f'ancService_{u}_{AncS}_{timestamp_str}', None)    
                    
                data_dict[t][u + '.TotalCosts'] = result_values.get(f'totalCosts_{u}_{timestamp_str}', None)
                data_dict[t][u + '.pCosts'] = result_values.get(f'pCosts_{u}_{timestamp_str}', None)
                data_dict[t][u + '.AS_Revenue'] = result_values.get(f'ancRevenue_{u}_{timestamp_str}', None)
                
                
                activePQLine = [result_values.get(f"PQLine_{u}_{timestamp_str}_{pql}", None) for pql in self.data['pqlData'][u]]
                active_line_segment = [pql for pql, pqlOnOff in zip(self.data['pqlData'][u], activePQLine) if pqlOnOff == 1]
                data_dict[t][u + '.ActiveLine'] = (active_line_segment[0] if active_line_segment else None)
        
        data_dict[self.sets['t'][0]]['TotalCost'] = self.result.objective_value()       
        
        # Convert to a DataFrame
        data = pd.DataFrame.from_dict(data_dict, orient='index')
        
    
        return data

    