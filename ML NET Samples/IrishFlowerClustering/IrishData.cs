using Microsoft.ML.Data;

namespace IrishFlowerClustering
{
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId; // contains the ID of the predicted cluster

        [ColumnName("Score")]
        public float[] Distances; // contains an array with squared Euclidean distances to the cluster centroids 
    }
}
