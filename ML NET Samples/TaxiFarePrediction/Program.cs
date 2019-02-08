using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;

namespace TaxiFarePrediction
{
    public class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6),
                }
            });

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);
        }

        /// <summary>
        /// This method:
        /// - loads the data
        /// - extracts and transforms the data
        /// - trains the model
        /// - saves the model as .zip file
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="dataPath"> Path to train dataset </param>
        /// <returns> The model </returns>
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId")) // Transforms text to numeric value
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            SaveModelAsFile(mlContext, model);

            return model;
        }

        /// <summary>
        /// This method saves the model as .zip file.
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }
        }

        /// <summary>
        /// This method:
        /// - loads the test dataset
        /// - creates the regression evaluator
        /// - evaluates the model and creates metrics
        /// - displays the metrics
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine();

            TestSinglePrediction(mlContext);
        }

        /// <summary>
        /// This method:
        /// - creates a single object of test data
        /// - predicts fare amount based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer loadedModel;
            using(var fs = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fs);
            }

            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VIS",
                RateCode = "6",
                PassengerCount = 5,
                TripTime = 7140,
                TripDistance = 3.75f,
                PaymentType = "CRD"
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
