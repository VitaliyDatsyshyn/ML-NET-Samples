using Microsoft.ML.Data;

namespace IrishFlowerClustering
{
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string FlowerType;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedCluster; 

        [ColumnName("Score")]
        public float[] Distances; // contains an array with squared Euclidean distances to the cluster centroids 
    }
}
