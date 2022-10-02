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

        public static double[] RunRandomForestClassifier(MLContext mlContext, IDataView trainingData, IDataView testingData, string labelName, int numberOfTrees, int numberOfLeaves)
        {
            var featuresArray = GetFeaturesArray(trainingData, labelName);
            var preprocessingPipeline = mlContext.Transforms.Concatenate("Features", featuresArray);
            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            FastForestBinaryTrainer.Options options = new FastForestBinaryTrainer.Options();
            options.LabelColumnName = labelName;
            options.FeatureColumnName = "Features";
            options.NumberOfTrees = numberOfTrees;
            options.NumberOfLeaves = numberOfLeaves;
            options.MinimumExampleCountPerLeaf = 5;
            options.FeatureFraction = 1.0;

            var trainer = mlContext.BinaryClassification.Trainers.FastForest(options);

            ITransformer model = trainer.Fit(preprocessedTrainingData);

            IDataView trainingPredictions = model.Transform(preprocessedTrainingData);
            var trainingMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(trainingPredictions, labelColumnName: labelName);
            IDataView testingPredictions = model.Transform(preprocessedTestingData);
            var testingMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testingPredictions, labelColumnName: labelName);

            double[] metrics = new double[4];
            metrics[0] = trainingMetrics.Accuracy;
            metrics[1] = testingMetrics.Accuracy;
            metrics[2] = trainingMetrics.F1Score;
            metrics[3] = testingMetrics.F1Score;
            return metrics;
        }

    	static void Main(string[] args)
        {
        	// args[0] - training data filename
            // args[1] - testing data filename
            // args[2] - machine learning task (regression/binary)
            // Random Forest parameters:
            //     args[3] - NumberOfTrees
            //     args[4] - NumberOfLeaves
            var mlContext = new MLContext(seed: 42);
            // data[0] - training subset
            // data[1] - testing subset
            IDataView[] data = LoadData(mlContext, args[0], args[1], args[2]);
            string labelName = "target";

        	var mainWatch = System.Diagnostics.Stopwatch.StartNew();
            double[] metrics;
            if (args[2] == "binary")
            {
                int numberOfTrees = Int32.Parse(args[3]);
                int numberOfLeaves = Int32.Parse(args[4]);
                metrics = RunRandomForestClassifier(mlContext, data[0], data[1], labelName, numberOfTrees, numberOfLeaves);
                mainWatch.Stop();
                Console.WriteLine("algorithm,all workflow time[ms],training accuracy,testing accuracy,training F1 score,testing F1 score");
                Console.WriteLine($"Random Forest Classification,{mainWatch.Elapsed.TotalMilliseconds},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}");
            }

        }
    }
}
