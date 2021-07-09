/* file: lin_reg_norm_eq_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    C++ example of multiple linear regression in the batch processing mode.
!
!    The program trains the multiple linear regression model on a training
!    datasetFileName with the normal equations method and computes regression
!    for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LINEAR_REGRESSION_NORM_EQ_BATCH"></a>
 * \example lin_reg_norm_eq_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::linear_regression;

typedef float FPType;

/* Input data set parameters */
string trainDatasetFileName = "data/DATASET_train.csv";
string testDatasetFileName  = "data/DATASET_test.csv";

const size_t nFeatures           = FEATURES; /* Number of features in training and testing data sets */
const size_t nDependentVariables = 1;  /* Number of dependent variables that correspond to each observation */

void trainAndTestModel();

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainAndTestModel();

    return 0;
}

void getMetrics(NumericTable * truthTable, NumericTable * predictionTable, FPType * metrics)
{
    size_t n = truthTable->getNumberOfRows();
    BlockDescriptor<FPType> truthBlock, predictionBlock;
    truthTable->getBlockOfRows(0, n, readWrite, truthBlock);
    predictionTable->getBlockOfRows(0, n, readWrite, predictionBlock);

    FPType * truth = truthBlock.getBlockPtr();
    FPType * prediction = predictionBlock.getBlockPtr();

    FPType mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += truth[i];
    mean /= n;

    FPType sqSum = 0.0;
    FPType mae = 0.0;
    FPType rmse = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sqSum += pow(truth[i] - mean, 2);
        mae += abs(truth[i] - prediction[i]);
        rmse += pow(truth[i] - prediction[i], 2);
    }
    FPType r2 = 1 - rmse / sqSum;
    mae /= n;
    rmse /= n;
    rmse = pow(rmse, 0.5);

    metrics[0] = mae;
    metrics[1] = rmse;
    metrics[2] = r2;
}

void trainAndTestModel()
{
    CsvDataSourceOptions csvOptions(CsvDataSourceOptions::allocateNumericTable | CsvDataSourceOptions::createDictionaryFromContext | CsvDataSourceOptions::parseHeader);

    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, csvOptions);
    NumericTablePtr trainData(new HomogenNumericTable<FPType>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainDependentVariables(new HomogenNumericTable<FPType>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainMergedData(new MergedNumericTable(trainData, trainDependentVariables));
    trainDataSource.loadDataBlock(trainMergedData.get());

    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, csvOptions);
    NumericTablePtr testData(new HomogenNumericTable<FPType>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr testGroundTruth(new HomogenNumericTable<FPType>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr testMergedData(new MergedNumericTable(testData, testGroundTruth));
    testDataSource.loadDataBlock(testMergedData.get());

    size_t trainingRows = trainData->getNumberOfRows();
    size_t testingRows = testData->getNumberOfRows();

    training::Batch<FPType> trainingAlgorithm;
    trainingAlgorithm.input.set(training::data, trainData);
    trainingAlgorithm.input.set(training::dependentVariables, trainDependentVariables);

    auto t0 = high_resolution_clock::now();
    trainingAlgorithm.compute();
    auto t1 = high_resolution_clock::now();
    duration<double, std::milli> ms0 = t1 - t0;
    double trainingTime = ms0.count();

    training::ResultPtr trainingResult = trainingAlgorithm.getResult();

    prediction::Batch<FPType> predictionAlgorithm;
    predictionAlgorithm.input.set(prediction::data, testData);
    predictionAlgorithm.input.set(prediction::model, trainingResult->get(training::model));

    auto t2 = high_resolution_clock::now();
    predictionAlgorithm.compute();
    auto t3 = high_resolution_clock::now();
    duration<double, std::milli> ms1 = t3 - t2;
    double predictionTime = ms1.count();

    NumericTablePtr testingPredictions = predictionAlgorithm.getResult()->get(prediction::prediction);

    prediction::Batch<FPType> predictionAlgorithm2;
    predictionAlgorithm2.input.set(prediction::data, trainData);
    predictionAlgorithm2.input.set(prediction::model, trainingResult->get(training::model));
    predictionAlgorithm2.compute();

    NumericTablePtr trainingPredictions = predictionAlgorithm2.getResult()->get(prediction::prediction);

    FPType trainingMetrics[3];
    FPType testingMetrics[3];

    getMetrics(trainDependentVariables.get(), trainingPredictions.get(), trainingMetrics);
    getMetrics(testGroundTruth.get(), testingPredictions.get(), testingMetrics);

    printf("%s\n", "Algorithm,Dataset,Training Rows,Testing Rows,Columns,Training MeanAbsoluteError,Training RootMeanSquaredError,Training RSquared,Testing MeanAbsoluteError,Testing RootMeanSquaredError,Testing RSquared,Training Time[ms],Prediction Time[ms]");
    printf("%s,%s,%lu,%lu,%lu,%f,%f,%f,%f,%f,%f,%f,%f\n", "LinearRegression", "DATASET", trainingRows, testingRows, nFeatures,
        trainingMetrics[0], trainingMetrics[1], trainingMetrics[2], testingMetrics[0], testingMetrics[1], testingMetrics[2], trainingTime, predictionTime);
}
