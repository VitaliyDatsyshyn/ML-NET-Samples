using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView trainData = mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            ITransformer model = BuildAndTrainModel(mlContext, trainData);

            Evaluate(mlContext, model);

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
            var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized") // Featurizing the text (Title and Description) columns into
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized")) // a numeric vector which is used by the ML algorithm.
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized")) // Combines all of the feature columns into the Features column, because, by default, learning algorithm processes features only from the Features column.
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Area"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent()) // By default, the values in Label column are considered as correct values to be predicted, so we Area column into Label column.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); // Map the Label to the value to return to its ordinal readable state.

            ITransformer model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        ///  This method:
        /// - loads the test data
        /// - creates the multiclass evaluator
        /// - evaluates model and create metrics
        /// - displays the metrics
        /// </summary>
        /// <param name="mlContext"> MLContext object </param>
        /// <param name="model"> The model </param>
        public static void Evaluate(MLContext mlContext, ITransformer model)
        {
            var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView)); // Computes the quality metrics.

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");    // Every sample-class pair contributes equally to the accuracy metric. 1 - the best result.
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");    // Minority classes are given equal weight as the larger classes. 1 - the best result.
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");          // That characterizes the accuracy of a classifier. 0 - the best result.
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}"); // 0 - the best result.
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(mlContext, model);
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
            PredictionEngine<GitHubIssue, IssuePrediction> predictionFunction = model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(mlContext);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing."
            };

            var predictionResults = predictionFunction.Predict(issue);

            Console.WriteLine($"\n=============== Single Prediction - Result: {predictionResults.Area} ===============\n");
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
            IEnumerable<GitHubIssue> issues = new[]
            {
                new GitHubIssue
                {
                    Title = "Garbage Collector bug",
                    Description = "GC is working bad."
                },
                new GitHubIssue
                {
                    Title = "Entity Framework crashes",
                    Description = "When connecting to the database, EF is crashing."
                }
            };

            ITransformer loadedModel;
            using(var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            IDataView issuesDataView = mlContext.Data.LoadFromEnumerable(issues);

            IDataView predictions = loadedModel.Transform(issuesDataView);

            IEnumerable<IssuePrediction> predictedResults = mlContext.Data.CreateEnumerable<IssuePrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            Console.WriteLine();

            foreach (var item in predictedResults)
            {
                Console.WriteLine($"\n=============== Result: {item.Area} ===============\n");
            }
        }
    }
}
