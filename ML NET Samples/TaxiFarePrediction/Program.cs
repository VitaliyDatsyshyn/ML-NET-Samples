using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace TaxiFarePrediction
{
    public class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView trainData = mlContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, separatorChar: ',', hasHeader: true);

            ITransformer model = BuildAndTrainModel(mlContext, trainData);

            Evaluate(mlContext, model);

            UseModelWithSingleItem(mlContext, model);

            UseLoadedModelWithBatchItems(mlContext);
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
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: "FareAmount", outputColumnName: "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId")) // Transforms text to numeric value
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(trainData);

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

            Console.WriteLine("The model is saved to {0}", _modelPath);
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
            IDataView testData = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, separatorChar: ',', hasHeader: true);

            var predictions = model.Transform(testData);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine();            
        }

        /// <summary>
        /// This method:
        /// - creates a single item of test data
        /// - predicts sentiment based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<TaxiTrip, TaxiTripFarePrediction> predictionFunction = model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VIS",
                RateCode = "6",
                PassengerCount = 5,
                TripTime = 7140,
                TripDistance = 3.75f,
                PaymentType = "CRD"
            };

            var predictedResults = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {predictedResults.FareAmount:0.####}");
            Console.WriteLine($"**********************************************************************");
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
            IEnumerable<TaxiTrip> taxiTrips = new[]
            {
                new TaxiTrip
                {
                    VendorId = "VIS",
                    RateCode = "6",
                    PassengerCount = 5,
                    TripTime = 7140,
                    TripDistance = 3.75f,
                    PaymentType = "CRD"
                },
                new TaxiTrip
                {
                    VendorId = "CMT",
                    RateCode = "3",
                    PassengerCount = 2,
                    TripTime = 1140,
                    TripDistance = 1.75f,
                    PaymentType = "CRD"
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            IDataView taxiTripsDataView = mlContext.Data.LoadFromEnumerable(taxiTrips);

            IDataView predictions = loadedModel.Transform(taxiTripsDataView);

            IEnumerable<TaxiTripFarePrediction> predictedResults = mlContext.Data.CreateEnumerable<TaxiTripFarePrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            Console.WriteLine();

            foreach (var item in predictedResults)
            {
                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Predicted fare: {item.FareAmount:0.####}");
                Console.WriteLine($"**********************************************************************");
            }
        }
    }
}
