import java.util.*;

public class EvaluateModel {
    private CaptionGenerator captionGenerator;
    
    public EvaluateModel(CaptionGenerator captionGenerator) {
        this.captionGenerator = captionGenerator;
    }
    
    public double computeBLEUScore(List<String> referenceCaptions, String predictedCaption) {
        List<List<String>> references = new ArrayList<>();
        for (String ref : referenceCaptions) {
            references.add(Arrays.asList(ref.split(" ")));
        }
        List<String> hypothesis = Arrays.asList(predictedCaption.split(" "));
        
        return calculateBLEU(references, hypothesis);
    }
    
    private double calculateBLEU(List<List<String>> references, List<String> hypothesis) {
        int[] nGramSizes = {1, 2, 3, 4};
        double score = 0.0;
        
        for (int n : nGramSizes) {
            double precision = calculateNGramPrecision(references, hypothesis, n);
            score += Math.log(precision + 1e-10); // Avoid log(0)
        }
        
        score = Math.exp(score / nGramSizes.length);
        double brevityPenalty = computeBrevityPenalty(references, hypothesis);
        
        return brevityPenalty * score;
    }
    
    private double calculateNGramPrecision(List<List<String>> references, List<String> hypothesis, int n) {
        Map<String, Integer> hypothesisNGrams = extractNGrams(hypothesis, n);
        int matched = 0;
        int total = hypothesisNGrams.values().stream().mapToInt(Integer::intValue).sum();
        
        for (List<String> reference : references) {
            Map<String, Integer> referenceNGrams = extractNGrams(reference, n);
            for (String nGram : hypothesisNGrams.keySet()) {
                if (referenceNGrams.containsKey(nGram)) {
                    matched += Math.min(hypothesisNGrams.get(nGram), referenceNGrams.get(nGram));
                }
            }
        }
        
        return total == 0 ? 0 : (double) matched / total;
    }
    
    private Map<String, Integer> extractNGrams(List<String> words, int n) {
        Map<String, Integer> nGrams = new HashMap<>();
        for (int i = 0; i <= words.size() - n; i++) {
            String nGram = String.join(" ", words.subList(i, i + n));
            nGrams.put(nGram, nGrams.getOrDefault(nGram, 0) + 1);
        }
        return nGrams;
    }
    
    private double computeBrevityPenalty(List<List<String>> references, List<String> hypothesis) {
        int hypLen = hypothesis.size();
        int refLen = references.stream().mapToInt(List::size).min().orElse(hypLen);
        
        return hypLen > refLen ? 1.0 : Math.exp(1 - (double) refLen / hypLen);
    }
}