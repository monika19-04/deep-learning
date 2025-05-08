import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

public class AnomalyDetector {
    public static void main(String[] args) {
        int seed = 123;
        int batchSize = 16;
        int numFeatures = 3;
        int epochs = 30;

        // Simulated normal data (close to [0, 0, 0])
        List<DataSet> trainingData = new ArrayList<>();
        Random rand = new Random(seed);
        for (int i = 0; i < 1000; i++) {
            double[] features = new double[]{
                rand.nextGaussian() * 0.5,
                rand.nextGaussian() * 0.5,
                rand.nextGaussian() * 0.5
            };
            INDArray input = Nd4j.create(features);
            trainingData.add(new DataSet(input, input));  // input = target (autoencoder)
        }

        var trainIter = new ListDataSetIterator<>(trainingData, batchSize);

        // Autoencoder network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(0.001))
            .list()
            .layer(new DenseLayer.Builder().nIn(numFeatures).nOut(4)
                .activation(Activation.RELU).build())
            .layer(new DenseLayer.Builder().nOut(2)
                .activation(Activation.RELU).build())  // bottleneck
            .layer(new DenseLayer.Builder().nOut(4)
                .activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nOut(numFeatures).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.fit(trainIter, epochs);

        // Test data: some normal, some anomalies
        INDArray[] testPoints = new INDArray[]{
            Nd4j.create(new double[]{0.1, -0.2, 0.05}),     // normal
            Nd4j.create(new double[]{5, 5, 5}),             // anomaly
            Nd4j.create(new double[]{-3, 2, 0}),            // anomaly
            Nd4j.create(new double[]{0.2, 0.1, -0.1})       // normal
        };

        double threshold = 1.0;  // reconstruction error threshold
        for (INDArray testPoint : testPoints) {
            INDArray output = model.output(testPoint);
            double error = testPoint.squaredDistance(output);
            System.out.println("Input: " + testPoint + " | Reconstruction error: " + error +
                (error > threshold ? " => Anomaly" : " => Normal"));
        }
    }
}
