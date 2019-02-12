using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace SentimentAnalysis
{
    public class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Separators = new char[] { '\t' },
                HasHeader = true,
                Column = new[]
                                                           {
                                                                new TextLoader.Column("Label", DataKind.Bool, 0),
                                                                new TextLoader.Column("SentimentText", DataKind.Text, 1)
                                                           },

            }
            );

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            Predict(mlContext, model);

            PredictWithModelLoadedFromFile(mlContext);
        }

        /// <summary>
        /// This method: 
        /// - loads the data
        /// - extracts and transforms the data
        /// - thains the model
        /// - predicts sentiments based on test data
        /// - returns the model
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="dataPath"> Path to data </param>
        /// <returns> The model </returns>
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath); //Load the data

            Console.WriteLine("=============== Create and Train the Model ===============");

            var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "SentimentText", outputColumnName: "Features")                                    // Extract and transform the data
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 40, numTrees: 40, minDatapointsInLeaves: 20)); // Choose a learning algorithm

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            var model = pipeline.Fit(dataView); // Create and train the model

            return model; // Return and save the model trained to use for evaluation
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
        public static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath); // Load the test dataset

            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

            var predictions = model.Transform(dataView); // Input features and return predictions

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label"); // Computes the quality metrics

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            SaveModelAsFile(mlContext, model);
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
        private static void Predict(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext); // Prediction function for individual examples

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "  \\  :: Bad MOVIE"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Toxic" : "Not toxic")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
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
        /// - creates batch test data
        /// - predicts sentiments based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        public static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This is a cool movie."
                },
                new SentimentData
                {
                    SentimentText = "BAD movie"
                }
            };

            // Load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine
            var sentimentStreamingDataView = mlContext.Data.ReadFromEnumerable(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxit (1) or nice (0)
            var predictedResults = mlContext.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

            // Combine the sentiment and prediction together to see the original comment with its predicted sentiment.
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not toxic")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
