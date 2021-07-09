#!/bin/bash

echo 'ML.NET part'
dotnet run abalone regression LinearRegression
dotnet run year_prediction_msd regression LinearRegression noheader

echo 'oneDAL part'
python run_dal.py
