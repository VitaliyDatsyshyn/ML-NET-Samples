using System;
using System.IO;
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
            var mlContext = new MLContext(seed: 0);

            TextLoader textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Separators = new char[] { ',' },
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3)
                }
            });

            IDataView dataView = textLoader.Read(_dataPath);

            var pipeline = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans("Features", clustersCount: 3));

            var model = pipeline.Fit(dataView);

            using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            var predictor = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);

            var setosa = new IrisData()
            {
                SepalLength = 0.1f,
                SepalWidth = 0.5f,
                PetalLength = 0.4f,
                PetalWidth = 0.2f
            };

            var prediction = predictor.Predict(setosa);

            Console.WriteLine($"Cluster ID: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }
}
