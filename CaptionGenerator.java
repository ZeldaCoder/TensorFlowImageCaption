import org.tensorflow.*;
import java.util.*;

public class CaptionGenerator {
    private DecoderWithAttention decoder;
    private Encoder encoder;
    private Map<Integer, String> wordMap;
    private int beamSize;
    
    public CaptionGenerator(DecoderWithAttention decoder, Encoder encoder, Map<Integer, String> wordMap, int beamSize) {
        this.decoder = decoder;
        this.encoder = encoder;
        this.wordMap = wordMap;
        this.beamSize = beamSize;
    }
    
    public String generateCaption(Tensor<TFloat32> image) {
        System.out.println("Generating caption using beam search...");
        
        Tensor<TFloat32> features = encoder.encodeImage(image);
        List<Integer> captionIndices = beamSearch(features);
        
        StringBuilder caption = new StringBuilder();
        for (int index : captionIndices) {
            caption.append(wordMap.getOrDefault(index, "<UNK>"))
                   .append(" ");
        }
        
        return caption.toString().trim();
    }
    
    private List<Integer> beamSearch(Tensor<TFloat32> features) {
        PriorityQueue<BeamEntry> queue = new PriorityQueue<>(Comparator.comparingDouble(e -> -e.probability));
        queue.add(new BeamEntry(new ArrayList<>(), 1.0));
        
        for (int step = 0; step < 20; step++) { // Max caption length
            PriorityQueue<BeamEntry> newQueue = new PriorityQueue<>(Comparator.comparingDouble(e -> -e.probability));
            
            while (!queue.isEmpty()) {
                BeamEntry entry = queue.poll();
                if (!entry.sequence.isEmpty() && entry.sequence.get(entry.sequence.size() - 1) == 0) {
                    newQueue.add(entry);
                    continue;
                }
                
                Map<Integer, Double> nextWordProbs = decoder.predictNextWordWithLSTM(features, entry.sequence);
                for (Map.Entry<Integer, Double> wordProb : nextWordProbs.entrySet()) {
                    List<Integer> newSequence = new ArrayList<>(entry.sequence);
                    newSequence.add(wordProb.getKey());
                    newQueue.add(new BeamEntry(newSequence, entry.probability * wordProb.getValue()));
                }
            }
            
            queue = new PriorityQueue<>(newQueue.stream().limit(beamSize).toList());
        }
        
        return queue.peek().sequence;
    }
    
    private static class BeamEntry {
        List<Integer> sequence;
        double probability;
        
        BeamEntry(List<Integer> sequence, double probability) {
            this.sequence = sequence;
            this.probability = probability;
        }
    }
}