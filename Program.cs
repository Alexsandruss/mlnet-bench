using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace mlnet_bench
{
    class Program
    {
        public static IDataView[] LoadData(MLContext mlContext, string dataset, string task, string label = "target", char separator = ',')
        {
            List<IDataView> dataList = new List<IDataView>();
            System.IO.StreamReader file = new System.IO.StreamReader($"data/{dataset}_train.csv");
            string header = file.ReadLine();
            file.Close();
            string[] headerArray = header.Split(separator);
            List<TextLoader.Column> columns = new List<TextLoader.Column>();
            foreach (string column in headerArray)
            {
                if (column == label)
                {
                    if (task == "binary")
                        columns.Add(new TextLoader.Column(column, DataKind.Boolean, Array.IndexOf(headerArray, column)));
                    else
                        columns.Add(new TextLoader.Column(column, DataKind.Single, Array.IndexOf(headerArray, column)));
                }
                else
                {
                    if (task == "recommendation")
                        columns.Add(new TextLoader.Column(column, DataKind.UInt32, Array.IndexOf(headerArray, column)));
                    else
                        columns.Add(new TextLoader.Column(column, DataKind.Single, Array.IndexOf(headerArray, column)));
                }
            }

            var loader = mlContext.Data.CreateTextLoader(
                separatorChar: separator,
                hasHeader: true,
                columns: columns.ToArray()
            );
            dataList.Add(loader.Load($"data/{dataset}_train.csv"));
            dataList.Add(loader.Load($"data/{dataset}_test.csv"));
            return dataList.ToArray();
        }

        public static string[] GetFeaturesArray(IDataView data, string labelName = "target")
        {
            List<string> featuresList = new List<string>();
            var nColumns = data.Schema.Count;
            var columnsEnumerator = data.Schema.GetEnumerator();
            for (int i = 0; i < nColumns; i++)
            {
                columnsEnumerator.MoveNext();
                if (columnsEnumerator.Current.Name != labelName)
                    featuresList.Add(columnsEnumerator.Current.Name);
            }

            return featuresList.ToArray();
        }

        public static List<double> EvaluateRegression(MLContext mlContext, IDataView predictions, string labelName)
        {
            List<double> metricsList = new List<double>();
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: labelName);
            metricsList.Add(metrics.MeanAbsoluteError);
            metricsList.Add(metrics.RootMeanSquaredError);
            metricsList.Add(metrics.RSquared);
            return metricsList;
        }

        public static List<double> EvaluateClassfication(MLContext mlContext, IDataView predictions, string labelName, bool isMulticlass = false)
        {
            List<double> metricsList = new List<double>();
            if (isMulticlass)
            {
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: labelName);
                metricsList.Add(metrics.LogLoss);
                metricsList.Add(metrics.MacroAccuracy);
            }
            else
            {
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: labelName);
                metricsList.Add(metrics.LogLoss);
                metricsList.Add(metrics.Accuracy);
            }
            return metricsList;
        }

        public static List<double> EvaluateClustering(MLContext mlContext, IDataView predictions, string labelName)
        {
            List<double> metricsList = new List<double>();
            var metrics = mlContext.Clustering.Evaluate(predictions, featureColumnName: "Features");
            metricsList.Add(metrics.AverageDistance);
            metricsList.Add(metrics.DaviesBouldinIndex);
            return metricsList;
        }

        public static double[] GetMetrics(MLContext mlContext, Func<IDataView, ITransformer> fitMethod, IDataView trainingData, IDataView testingData, string task, string labelName = "target")
        {
            var watch1 = System.Diagnostics.Stopwatch.StartNew();
            ITransformer model = fitMethod(trainingData);
            watch1.Stop();
            var elapsedMs1 = watch1.Elapsed.TotalMilliseconds;

            var watch2 = System.Diagnostics.Stopwatch.StartNew();
            IDataView testingPredictions = model.Transform(testingData);
            watch2.Stop();
            var elapsedMs2 = watch2.Elapsed.TotalMilliseconds;
            IDataView trainingPredictions = model.Transform(trainingData);

            if (task == "regression" || task == "recommendation")
            {
                var trainingMetrics = EvaluateRegression(mlContext, trainingPredictions, labelName);
                var testingMetrics = EvaluateRegression(mlContext, testingPredictions, labelName);

                double[] metrics = new double[] {
                    trainingMetrics[0], trainingMetrics[1], trainingMetrics[2], testingMetrics[0], testingMetrics[1], testingMetrics[2], elapsedMs1, elapsedMs2
                };

                return metrics;
            }
            else if (task == "binary" || task == "multi")
            {
                var trainingMetrics = EvaluateClassfication(mlContext, trainingPredictions, labelName, task == "multi");
                var testingMetrics = EvaluateClassfication(mlContext, testingPredictions, labelName, task == "multi");

                double[] metrics = new double[] {
                    trainingMetrics[0], trainingMetrics[1], testingMetrics[0], testingMetrics[1], elapsedMs1, elapsedMs2
                };
                return metrics;
            }
            else if (task == "clustering")
            {
                var trainingMetrics = EvaluateClustering(mlContext, trainingPredictions, labelName);
                var testingMetrics = EvaluateClustering(mlContext, testingPredictions, labelName);

                double[] metrics = new double[] {
                    trainingMetrics[0], trainingMetrics[1], testingMetrics[0], testingMetrics[1], elapsedMs1, elapsedMs2
                };

                return metrics;
            }
            return new double[]{};
        }

        public static double[] RunOLSRegression(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target")
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray);
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            var trainer = mlContext.Regression.Trainers.Ols(labelColumnName: labelName, featureColumnName: "Features");

            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "regression", labelName);
        }

        public static double[] RunBinaryLBFGSLogReg(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target")
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray);
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName: labelName,
                featureColumnName: "Features",
                l1Regularization: 1,
                l2Regularization: 1
            );

            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "binary", labelName);
        }

        public static double[] RunMultiLBFGSLogReg(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target")
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray).Append(mlContext.Transforms.Conversion.MapValueToKey(labelName));
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                labelColumnName: labelName,
                featureColumnName: "Features",
                l1Regularization: 1,
                l2Regularization: 1
            );

            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "multi", labelName);
        }

        public static double[] RunNaiveBayes(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target")
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray).Append(mlContext.Transforms.Conversion.MapValueToKey(labelName));
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            var trainer = mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: labelName, featureColumnName: "Features");
            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "multi", labelName);
        }

        public static double[] RunKMeans(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target", int nClusters = 5)
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray);
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: nClusters);

            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "clustering", labelName);
        }

        public static double[] RunMatrixFactorization(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName = "target", int rank = 10, int nIterations = 10)
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray).Append(mlContext.Transforms.Conversion.MapValueToKey("f0")).Append(mlContext.Transforms.Conversion.MapValueToKey("f1"));
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);
            var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(labelColumnName: labelName, matrixColumnIndexColumnName: "f0", matrixRowIndexColumnName: "f1", numberOfIterations: nIterations, approximationRank: rank);

            return GetMetrics(mlContext, trainer.Fit, preprocessedTrainingData, preprocessedTestingData, "recommendation", labelName);
        }

        public static void PrintMetrics(string dataset, string task, string algorithm, IDataView trainingData, IDataView testingData, double[] metrics, bool printHeader = true)
        {
            if (printHeader)
            {
                if (task == "regression")
                {
                    Console.WriteLine("Algorithm,Dataset,Training Rows,Testing Rows,Columns,Training MeanAbsoluteError,Training RootMeanSquaredError,Training RSquared," +
                        "Testing MeanAbsoluteError,Testing RootMeanSquaredError,Testing RSquared,Training Time[ms],Prediction Time[ms]");
                }
                else if (task == "binary" || task == "multi")
                {
                    Console.WriteLine("Algorithm,Dataset,Training Rows,Testing Rows,Columns,Training LogLoss,Training Accuracy,Testing LogLoss,Testing Accuracy,Training Time[ms],Prediction Time[ms]");
                }
                else if (task == "clustering")
                {
                    Console.WriteLine("Algorithm,Dataset,Training Rows,Testing Rows,Columns,Training AverageDistance,Training DaviesBouldinIndex,Testing AverageDistance,Testing DaviesBouldinIndex,Training Time[ms],Prediction Time[ms]");
                }
                else if (task == "recommendation")
                {
                    Console.WriteLine("Algorithm,Dataset,Training Rows,Testing Rows,Columns,Training MeanAbsoluteError,Training RootMeanSquaredError,Training RSquared," +
                        "Testing MeanAbsoluteError,Testing RootMeanSquaredError,Testing RSquared,Training Time[ms],Prediction Time[ms]");
                }
            }
            string output;
            if (task == "recommendation")
            {
                output = $"{algorithm},{dataset},{trainingData.GetColumn<uint>("f0").Count()},{testingData.GetColumn<uint>("f0").Count()},{testingData.Schema.Count - 1}";
            }
            else
            {
                output = $"{algorithm},{dataset},{trainingData.GetColumn<float>("f0").Count()},{testingData.GetColumn<float>("f0").Count()},{testingData.Schema.Count - 1}";
            }
            for (int i = 0; i < metrics.Length; i++)
            {
                output += $",{metrics[i]}";
            }
            Console.WriteLine(output);
        }

        static void Main(string[] args)
        {
            // args[0] - dataset prefix
            // args[1] - task (regression/binary/multi/clustering/recommendation/decomposition)
            // args[2] - algorithm
            // args[3] - number of clusters(KMeans) / components(PCA) / iterations(LinearSVM) / rank(MatrixFactorization)
            // args[4] - number of iterations(MatrixFactorization)
            var mlContext = new MLContext(seed: 42);
            // data[0] - training subset
            // data[1] - testing subset
            IDataView[] data = LoadData(mlContext, args[0], args[1]);

            bool printHeader = !(Array.Exists(args, element => element == "noheader"));
            if (args[2] == "LinearRegression")
            {
                double[] metrics = RunOLSRegression(mlContext, data[0], data[1]);
                PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
            }
            else if (args[2] == "LBFGSLogisticRegression")
            {
                double[] metrics = null;
                if (args[1] == "binary")
                {
                    metrics = RunBinaryLBFGSLogReg(mlContext, data[0], data[1]);
                    PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
                }
                else
                {
                    metrics = RunMultiLBFGSLogReg(mlContext, data[0], data[1]);
                    PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
                }
            }
            else if (args[2] == "NaiveBayes")
            {
                double[] metrics = RunNaiveBayes(mlContext, data[0], data[1]);
                PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
            }
            else if (args[2] == "KMeans")
            {
                double[] metrics = RunKMeans(mlContext, data[0], data[1], nClusters: Int32.Parse(args[3]));
                PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
            }
            else if (args[2] == "MatrixFactorization")
            {
                double[] metrics = RunMatrixFactorization(mlContext, data[0], data[1]);
                PrintMetrics(args[0], args[1], args[2], data[0], data[1], metrics, printHeader);
            }
        }
    }
}
