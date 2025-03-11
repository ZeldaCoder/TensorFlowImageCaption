import org.tensorflow.*;
import java.util.*;

public class DecoderWithAttention {
    private int hiddenSize;
    private int vocabSize;
    private LSTM lstm;
    private Attention attention;
    private Dense outputLayer;
    
    public DecoderWithAttention(int hiddenSize, int vocabSize) {
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.lstm = new LSTM(hiddenSize);
        this.attention = new Attention(hiddenSize);
        this.outputLayer = new Dense(vocabSize); // Fully connected output layer
    }
    
    public Map<Integer, Double> predictNextWordWithLSTM(Tensor<TFloat32> features, List<Integer> sequence) {
        Tensor<TFloat32> contextVector = attention.apply(features, sequence);
        Tensor<TFloat32> lstmOutput = lstm.forward(contextVector, sequence);
        Tensor<TFloat32> logits = outputLayer.apply(lstmOutput);
        
        return softmax(logits);
    }
    
    private Map<Integer, Double> softmax(Tensor<TFloat32> logits) {
        float[] values = logits.copyTo(new float[vocabSize]);
        double sumExp = Arrays.stream(values).map(Math::exp).sum();
        Map<Integer, Double> probabilities = new HashMap<>();
        
        for (int i = 0; i < vocabSize; i++) {
            probabilities.put(i, Math.exp(values[i]) / sumExp);
        }
        
        return probabilities;
    }
}