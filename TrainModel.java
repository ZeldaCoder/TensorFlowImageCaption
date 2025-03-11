import org.tensorflow.*;
import java.util.*;

public class TrainModel {
    private Encoder encoder;
    private DecoderWithAttention decoder;
    private Optimizer optimizer;
    private LossFunction lossFunction;
   
    public TrainModel(Encoder encoder, DecoderWithAttention decoder) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.optimizer = new AdamOptimizer();
        this.lossFunction = new CrossEntropyLoss();
    }
   
    public void train(List<TrainingExample> trainingData, int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double epochLoss = 0.0;
            for (TrainingExample example : trainingData) {
                epochLoss += trainStep(example);
            }
            System.out.println("Epoch " + epoch + " Loss: " + (epochLoss / trainingData.size()));
        }
    }
   
    private double trainStep(TrainingExample example) {
        Tensor<TFloat32> imageFeatures = encoder.encodeImage(example.getImage());
        List<Integer> targetSequence = example.getCaption();
       
        optimizer.zeroGrad();
       
        double totalLoss = 0.0;
        List<Integer> generatedSequence = new ArrayList<>();
       
        for (int t = 0; t < targetSequence.size() - 1; t++) {
            Map<Integer, Double> predictions = decoder.predictNextWordWithLSTM(imageFeatures, generatedSequence);
            int targetWord = targetSequence.get(t + 1);
            double loss = lossFunction.computeLoss(predictions, targetWord);
            totalLoss += loss;
        }
       
        optimizer.step();
        return totalLoss;
    }
}