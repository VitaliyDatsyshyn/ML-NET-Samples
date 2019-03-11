using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace SentimentAnalysis
{
    public class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            // Create ML.NET context/local environment - allows you to add steps in order to keep everything together 
            // during the learning process.  
            //Create ML Context with seed for repeatable/deterministic results
            MLContext mlContext = new MLContext();

            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);            

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model);

            UseLoadedModelWithBatchItems(mlContext);
        }

        /// <summary>
        /// This method:
        /// - loads the data
        /// - split the loaded dataset into train and test datasets
        /// </summary>
        /// <param name="mlContext"> MLContexxt object </param>
        /// <returns> The split train and test datasets </returns>
        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        /// <summary>
        /// This method:
        /// - extracts and transforms the data
        /// - trains the model
        /// - predicts sentiment based on test data
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="splitTrainSet"> Dataset for training </param>
        /// <returns> The model </returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Create a flexible pipeline (composed by a chain of estimators) for creating/training the model.
            // This is used to format and clean the data.  
            // Convert the text column to numeric vectors (Features column)
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                // Adds a FastTreeBinaryClassificationTrainer, the decision tree learner for this project 
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            // Create and train the model based on the dataset that has been loaded, transformed.
            var model = pipeline.Fit(splitTrainSet);

            return model;
        }

        /// <summary>
        /// This method:
        /// - loads the test dataset
        /// - creates the binary evaluator
        /// - evaluates the model and create metrics
        /// - displays the metrics
        /// </summary>
        /// <param name="mlContext">MLContext object </param>
        /// <param name="model"> The model </param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            
            //Take the data in, make transformations, output the data. 
            var predictions = model.Transform(splitTestSet);

            // BinaryClassificationContext.Evaluate returns a BinaryClassificationEvaluator.CalibratedResult
            // that contains the computed overall metrics.
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label"); // Computes the quality metrics

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            // Save the new model to .ZIP file
            SaveModelAsFile(mlContext, model);
        }

        /// <summary>
        /// This method saves the model as .zip file.
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            Console.WriteLine($"The model is saved to {_modelPath}");
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
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
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
            // Adds some comments to test the trained model's predictions.
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti"
                }
            };

            ITransformer loadedModel;
            using (FileStream stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Load test data.
            IDataView sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = loadedModel.Transform(sentimentStreamingDataView);

            // use model to predict whether comment data is Positive or Negative
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            Console.WriteLine();

            //Builds pairs of (sentiment, prediction)
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            foreach((SentimentData sentiment, SentimentPrediction prediction) item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
