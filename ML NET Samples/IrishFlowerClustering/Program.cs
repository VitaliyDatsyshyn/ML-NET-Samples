using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace IrishFlowerClustering
{
    public class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            IDataView trainData = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, separatorChar: ',');

            ITransformer model = BuildAndTrainModel(mlContext, trainData);

            UseModelWithSingleItem(mlContext, model);

            UseLoadedModelWithBatchItems(mlContext);
        }

        /// <summary>
        /// This method:
        /// - extracts and transforms the data
        /// - trains the model
        /// - predicts area based on training data
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="trainData"> Dataset for training </param>
        /// <returns> The model </returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "FlowerType")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.Clustering.Trainers.KMeans(clustersCount: 3));


            var model = pipeline.Fit(trainData);

            SaveModelAsFile(mlContext, model);

            return model;
        }

        /// <summary>
        /// This method saves model as .zip file.
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        public static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        /// <summary>
        /// This method:
        /// - creates a single comment of test data
        /// - predicts sentiment based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<IrisData, ClusterPrediction> predictionFunction = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);

            IrisData item = new IrisData()
            {
                PetalLength = 5.3f,
                PetalWidth = 3.1f,
                SepalLength = 1.6f,
                SepalWidth = 0.3f
            };

            var predictionResults = predictionFunction.Predict(item);

            Console.WriteLine($"Predicted iris type: {predictionResults.PredictedCluster}");
            Console.WriteLine($"Distances: {string.Join(" ", predictionResults.Distances)}");
        }

        /// <summary>
        /// This method:
        /// - creates batch test data
        /// - predicts sentiment based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {
            IEnumerable<IrisData> flowers = new[]
            {
                new IrisData
                {
                    PetalLength = 5.3f,
                    PetalWidth = 3.1f,
                    SepalLength = 1.6f,
                    SepalWidth = 0.3f
                },
                new IrisData
                {
                    PetalLength = 3.3f,
                    PetalWidth = 1.1f,
                    SepalLength = 0.6f,
                    SepalWidth = 0.9f
                }
            };

            ITransformer loadedModel;
            using(var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            IDataView flowersDataView = mlContext.Data.LoadFromEnumerable(flowers);

            IDataView predictons = loadedModel.Transform(flowersDataView);

            IEnumerable<ClusterPrediction> predictedResults = mlContext.Data.CreateEnumerable<ClusterPrediction>(predictons, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            Console.WriteLine();

            foreach (var item in predictedResults)
            {
                Console.WriteLine($"Predicted iris type: {item.PredictedCluster}");
                Console.WriteLine($"Distances: {string.Join(" ", item.Distances)}");
            }
        }
    }
}
