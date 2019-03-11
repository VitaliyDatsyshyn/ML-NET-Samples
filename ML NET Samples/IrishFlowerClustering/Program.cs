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

            //using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            //{
            //    mlContext.Model.Save(model, fs);
            //}
            
            

            double x1 = 5.3;
            double x2 = 3.1;
            double x3 = 1.6;
            double x4 = 0.3;


            var setosa = new IrisData()
            {
                SepalLength = (float) x1,
                SepalWidth = (float)x2,
                PetalLength = (float)x3,
                PetalWidth = (float)x4
            };

            Console.WriteLine(setosa.SepalLength);
            Console.WriteLine(setosa.SepalWidth);
            Console.WriteLine(setosa.PetalLength);
            Console.WriteLine(setosa.PetalWidth);

            var predictor = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);
            var prediction = predictor.Predict(setosa);

            Console.WriteLine($"Cluster ID: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

            
        }
    }
}
