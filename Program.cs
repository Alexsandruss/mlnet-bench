using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Newtonsoft.Json;

namespace mlnet_bench
{
    class Program
    {
        public static IDataView[] LoadData(
            MLContext mlContext, string trainingFile, string testingFile,
            string task, string label = "target", char separator = ',')
        {
            List<IDataView> dataList = new List<IDataView>();
            System.IO.StreamReader file = new System.IO.StreamReader(trainingFile);
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
                    columns.Add(new TextLoader.Column(column, DataKind.Single, Array.IndexOf(headerArray, column)));
                }
            }

            var loader = mlContext.Data.CreateTextLoader(
                separatorChar: separator,
                hasHeader: true,
                columns: columns.ToArray()
            );
            dataList.Add(loader.Load(trainingFile));
            dataList.Add(loader.Load(testingFile));
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

        public static IDataView[] PreprocessData(MLContext mlContext, string trainingFile, string testingFile,
            string task, string labelName, string featuresName = "Features")
        {
            IDataView[] data = LoadData(mlContext, trainingFile, testingFile, task, labelName);
            string[] featuresArray = GetFeaturesArray(data[0], labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate(featuresName, featuresArray);
            var pipeline = preprocessingPipeline.Fit(data[0]);
            var preprocessedTrainingData = pipeline.Transform(data[0]);
            var preprocessedTestingData = pipeline.Transform(data[1]);

            List<IDataView> dataList = new List<IDataView>();
            dataList.Add(preprocessedTrainingData);
            dataList.Add(preprocessedTestingData);

            return dataList.ToArray();
        }

        public static double[] EvaluateBinaryClassification(MLContext mlContext, ITransformer model, IDataView trainingData, IDataView testingData, string labelName)
        {
            IDataView trainingPredictions = model.Transform(trainingData);
            var trainingMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(trainingPredictions, labelColumnName: labelName);
            IDataView testingPredictions = model.Transform(testingData);
            var testingMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testingPredictions, labelColumnName: labelName);

            double[] metrics = new double[4];
            metrics[0] = trainingMetrics.Accuracy;
            metrics[1] = testingMetrics.Accuracy;
            metrics[2] = trainingMetrics.F1Score;
            metrics[3] = testingMetrics.F1Score;
            return metrics;
        }

        public static double[] EvaluateRegression(MLContext mlContext, ITransformer model, IDataView trainingData, IDataView testingData, string labelName)
        {
            IDataView trainingPredictions = model.Transform(trainingData);
            var trainingMetrics = mlContext.Regression.Evaluate(trainingPredictions, labelColumnName: labelName);
            IDataView testingPredictions = model.Transform(testingData);
            var testingMetrics = mlContext.Regression.Evaluate(testingPredictions, labelColumnName: labelName);

            double[] metrics = new double[4];
            metrics[0] = trainingMetrics.RootMeanSquaredError;
            metrics[1] = testingMetrics.RootMeanSquaredError;
            metrics[2] = trainingMetrics.RSquared;
            metrics[3] = testingMetrics.RSquared;
            return metrics;
        }

        public static double[] RunRandomForestClassification(MLContext mlContext, IDataView trainingData, IDataView testingData,
            int numberOfTrees, int numberOfLeaves, string labelName, string featuresName = "Features")
        {
            FastForestBinaryTrainer.Options options = new FastForestBinaryTrainer.Options();
            options.LabelColumnName = labelName;
            options.FeatureColumnName = featuresName;
            options.NumberOfTrees = numberOfTrees;
            options.NumberOfLeaves = numberOfLeaves;
            options.MinimumExampleCountPerLeaf = 5;
            options.FeatureFraction = 1.0;

            var trainer = mlContext.BinaryClassification.Trainers.FastForest(options);

            ITransformer model = trainer.Fit(trainingData);

            return EvaluateBinaryClassification(mlContext, model, trainingData, testingData, labelName);
        }

        public static double[] RunLogisticRegression(MLContext mlContext, IDataView trainingData, IDataView testingData,
            int numberOfIterations, string labelName, string featuresName = "Features")
        {
            var options = new LbfgsLogisticRegressionBinaryTrainer.Options();
            options.LabelColumnName = labelName;
            options.MaximumNumberOfIterations = numberOfIterations;
            options.L1Regularization = 0.01f;
            options.L2Regularization = 0.01f;

            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(options);

            ITransformer model = trainer.Fit(trainingData);

            return EvaluateBinaryClassification(mlContext, model, trainingData, testingData, labelName);
        }

        public static double[] RunRandomForestRegression(MLContext mlContext, IDataView trainingData, IDataView testingData,
            int numberOfTrees, int numberOfLeaves, string labelName, string featuresName = "Features")
        {
            FastForestRegressionTrainer.Options options = new FastForestRegressionTrainer.Options();
            options.LabelColumnName = labelName;
            options.FeatureColumnName = featuresName;
            options.NumberOfTrees = numberOfTrees;
            options.NumberOfLeaves = numberOfLeaves;
            options.MinimumExampleCountPerLeaf = 5;
            options.FeatureFraction = 1.0;

            var trainer = mlContext.Regression.Trainers.FastForest(options);

            ITransformer model = trainer.Fit(trainingData);

            return EvaluateRegression(mlContext, model, trainingData, testingData, labelName);
        }

        public static double[] RunOLSRegression(MLContext mlContext, IDataView trainingData, IDataView testingData,
            string labelName, string featuresName = "Features")
        {
            OlsTrainer.Options options = new OlsTrainer.Options();
            options.LabelColumnName = labelName;
            options.FeatureColumnName = featuresName;

            var trainer = mlContext.Regression.Trainers.Ols(options);

            ITransformer model = trainer.Fit(trainingData);

            return EvaluateRegression(mlContext, model, trainingData, testingData, labelName);
        }

        static void Main(string[] args)
        {
            // args[0] - training data filename
            // args[1] - testing data filename
            // args[2] - label name
            // args[3] - machine learning task (regression, binary)
            // args[4] - machine learning algorithm (RandomForest, OLS, LR)
            // Random Forest parameters:
            //     args[5] - NumberOfTrees
            //     args[6] - NumberOfLeaves
            // Logistic Regression parameters:
            //     args[5] - Number of iterations

            var mlContext = new MLContext(seed: 42);
            // data[0] - training subset
            // data[1] - testing subset
            string labelName = args[2];
            IDataView[] data = PreprocessData(mlContext, args[0], args[1], args[3], labelName);

            var mainWatch = System.Diagnostics.Stopwatch.StartNew();
            double[] metrics;
            if (args[4] == "RF")
            {
                int numberOfTrees = Int32.Parse(args[5]);
                int numberOfLeaves = Int32.Parse(args[6]);
                if (args[3] == "binary")
                {

                    metrics = RunRandomForestClassification(mlContext, data[0], data[1], numberOfTrees, numberOfLeaves, labelName);
                    mainWatch.Stop();
                    Console.WriteLine("algorithm,workload time[ms],training accuracy,testing accuracy,training F1 score,testing F1 score");
                    Console.WriteLine($"Random Forest Binary,{mainWatch.Elapsed.TotalMilliseconds},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}");
                }
                else
                {
                    metrics = RunRandomForestRegression(mlContext, data[0], data[1], numberOfTrees, numberOfLeaves, labelName);
                    mainWatch.Stop();
                    Console.WriteLine("algorithm,workload time[ms],training RMSE,testing RMSE,training R2 score,testing R2 score");
                    Console.WriteLine($"Random Forest Regression,{mainWatch.Elapsed.TotalMilliseconds},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}");
                }
            }
            else if (args[4] == "OLS")
            {
                metrics = RunOLSRegression(mlContext, data[0], data[1], labelName);
                mainWatch.Stop();
                Console.WriteLine("algorithm,workload time[ms],training RMSE,testing RMSE,training R2 score,testing R2 score");
                Console.WriteLine($"OLS Regression,{mainWatch.Elapsed.TotalMilliseconds},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}");
            }
            else if (args[4] == "LR")
            {
                int numberOfIterations = Int32.Parse(args[5]);
                metrics = RunLogisticRegression(mlContext, data[0], data[1], numberOfIterations, labelName);
                mainWatch.Stop();
                Console.WriteLine("algorithm,workload time[ms],training accuracy,testing accuracy,training F1 score,testing F1 score");
                Console.WriteLine($"Logistic Regression,{mainWatch.Elapsed.TotalMilliseconds},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}");
            }
        }
    }
}
