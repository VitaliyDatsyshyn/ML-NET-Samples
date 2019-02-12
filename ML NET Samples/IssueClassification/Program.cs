using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.Data.DataView;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine; // Used for single predictions.
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            _trainingDataView = _mlContext.Data.CreateTextLoader<GitHubIssue>(hasHeader: true).Read(_trainDataPath);

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate();

            PredictIssue();
        }

        /// <summary>
        /// This method extracts and transforms the data. 
        /// </summary>
        /// <returns> The processing pipeline </returns>
        public static EstimatorChain<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")  // By default, the values in Label column are considered as correct values to be predicted, so we Area column into Label column.
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName:"TitleFeaturized"))             // Featurizing the text (Title and Description) columns into
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized")) // a numeric vector which is used by the ML algorithm.
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized")) // Combines all of the feature columns into the Features column, because, by default, learning algorithm processes features only from the Features column.
                .AppendCacheCheckpoint(_mlContext); // Cache the DataView so when we iterate over the data multiple times using the cache might get better performance.

            return pipeline;
        }

        /// <summary>
        /// This method:
        /// - creates the training algorithm class
        /// - trains the model
        /// - predicts area based on training data
        /// - saves the model to a .zip file
        /// </summary>
        /// <param name="trainingDataView"> IDataView object used to process the training dataset </param>
        /// <param name="pipeline"> For the processing pipeline </param>
        /// <returns> The model </returns>
        public static EstimatorChain<KeyToValueMappingTransformer> BuildAndTrainModel(IDataView trainingDataView, EstimatorChain<ITransformer> pipeline)
        {

            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features")) // Append learning alogoritm to the pipeline
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); // Map the Label to the value to return to its ordinal readable state.

            _trainedModel = trainingPipeline.Fit(trainingDataView); // Returns a model to use for predictions.

            _predEngine = _trainedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_mlContext); // For predictions on individual examples.

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue); // Create single prediction.

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        /// <summary>
        /// This method:
        /// - loads the test data
        /// - creates the multiclass evaluator
        /// - evaluates model and create metrics
        /// - displays the metrics
        /// </summary>
        public static void Evaluate()
        {
            var testDataView = _mlContext.Data.CreateTextLoader<GitHubIssue>(hasHeader: true).Read(_testDataPath);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView)); // Computes the quality metrics.

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");    // Every sample-class pair contributes equally to the accuracy metric. 1 - the best result.
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");    // Minority classes are given equal weight as the larger classes. 1 - the best result.
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");          // That characterizes the accuracy of a classifier. 0 - the best result.
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}"); // 0 - the best result.
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsZipFile(_mlContext, _trainedModel);
        }

        /// <summary>
        /// This method:
        /// - creates a single issue of test data
        /// - predicts Area based on test data
        /// - combines test data and predictions for reporting
        /// - displays the predicted results
        /// </summary>
        private static void PredictIssue()
        {
            ITransformer loadedModel;
            using(var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = _mlContext.Model.Load(stream);
            }

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing."
            };

            _predEngine = loadedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_mlContext);

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        /// <summary>
        /// This method saves model as .zip file.
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        public static void SaveModelAsZipFile(MLContext mlContext, ITransformer model)
        {
            using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
