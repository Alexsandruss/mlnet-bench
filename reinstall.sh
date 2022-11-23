#!/bin/bash

cd ../machinelearning
./build.sh
cd ../mlnet-bench
dotnet build
