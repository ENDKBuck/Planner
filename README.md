#  Dispatcher v0.1
The OPTINEERING Dispatch Model allows the user to maxmimize the value creation of assets by utlizing all the energy markets incl. Ancillary services. 
The Dispatcher optimises total costs while considering all necessary constraints suchs as heat demand, temperature, ramping and much more.


## How to use this repo
1. Install the packages in `requirements.txt`.
2. Open Excel file to check all input data 
   You can skip Excel check step and just run main.py.
3. Run `python -m main`.
   This runs the  fetching of data (tariffs, prices etc. from energidataservice.dk) and runs the optimization (Dispatcher) for the chosen data.
   The results are saved as in the Excel 'Results' sheet and also input data is saved under Excel sheet 'TimeSeries'


#Script steps

## Gathering input data

1. Loading of Excel to identify plant input and what to simulate
2. Fetching of price data from www.energidataservice.dk via EDS_caller()' a class from `EDS_extract.py`
3. Heat input is currecntly just arbitraty and is to be changed with actual heat demand
4. Some manual input are inserted in script.

Eventually the Timeseries contains all needed data to do the optimization.  

## Creation of Unit data 
Unit data is treated via the `Helper_functions.py` and basically ensures that the optimization is linearly following the operational points decided by the operator. 
As an example if p1(power,heat, fuel) = 1,1,1 and p2(power,heat,fuel) = 3,3,3 then the helper function creates them as lineset via interpolation, hence getting linear equations between the points.  

##  Dispatcher
The dispatcher splits the input data up into batches (if needed) to optimize over smaller timeframes. The amount of batches is defined in the Excel Sheet under "main". The loop overlaps to ensure that optimization close to the end is used (e.g. depletion of storage due to "end of time").

# Data structure
Both input data and results have the same structure and depends on timestamps as index series.

## Input data
The input data needed depends on the active markets defined in the excel sheet. To run script with minimun (no ancillary service) the following data is needed.
- Unit input (float)
- Elspot (float)
- Heat Demand (float)
- Avaiablity of the unit (0-1)
- Tariffs

## Results
The dataframe, df, contains all the results which is stored in the excel sheet. It contains a lot of operation information.

## DA Bud to Excel
Early start on creating a bid entry for spot market. This is not developed

## Plots
Currently containing three types of plots to show results. 
1. Sold capacity to ancillary service market
2. Production plan and power balance
3. Operational points (mostly to validate heat production)