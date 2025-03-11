import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.List;

public class BeamSearchDecoder {

    public static List<String> decodeWithBeamSearch(Tensor<Float> imageFeatures, int beamWidth, String[] vocab) {
        // Initialize the RNN model (LSTM) for caption generation
        Graph graph = new Graph();
        byte[] graphBytes = Files.readAllBytes(Paths.get("path/to/lstm_model.pb"));
        graph.importGraphDef(graphBytes);

        // Initialize the session
        try (Session session = new Session(graph)) {

            // Initialize the beam search structures
            List<BeamSearchNode> beams = new ArrayList<>();
            beams.add(new BeamSearchNode("", 0.0f, imageFeatures));  // Start with an empty sequence

            List<String> bestCaption = new ArrayList<>();
            int maxLength = 50;  // Maximum length of caption (can be adjusted)

            // Beam search loop for maxLength steps
            for (int step = 0; step < maxLength; step++) {

                List<BeamSearchNode> allCandidates = new ArrayList<>();

                // Expand each sequence in the beam
                for (BeamSearchNode beam : beams) {
                    String[] previousTokens = beam.sequence.split(" "); // Split the current sequence into tokens

                    // Convert the tokens into tensor input for the RNN model
                    Tensor<Float> tokenInput = convertTokensToTensor(previousTokens, vocab);

                    // Feed the image features and previous token into the RNN
                    Tensor<Float> output = session.runner()
                            .feed("input_image_features", beam.imageFeatures)
                            .feed("input_token", tokenInput)
                            .fetch("output_caption_tensor")
                            .run().get(0);

                    // Get the top-k predictions for the next word
                    List<String> topKWords = getTopKWords(output, vocab, beamWidth);

                    // Create new sequences for each of the top-k words
                    for (String word : topKWords) {
                        float score = beam.score + Math.log(getWordProbability(word, output));  // Log probability for scoring
                        String newSequence = beam.sequence + " " + word;
                        allCandidates.add(new BeamSearchNode(newSequence, score, beam.imageFeatures));
                    }
                }

                // Sort all candidates by score (higher is better) and keep top-k sequences
                beams = getTopKSequences(allCandidates, beamWidth);
            }

            // Choose the best sequence as the final caption
            bestCaption = beams.get(0).getSequenceTokens();
            return bestCaption;
        }
    }

    // Helper function to convert tokens to Tensor for RNN input
    private static Tensor<Float> convertTokensToTensor(String[] tokens, String[] vocab) {
        float[] tokenIds = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            tokenIds[i] = getTokenId(tokens[i], vocab); // Map tokens to their corresponding ID
        }
        return Tensor.create(new long[]{1, tokens.length}, Float.class, tokenIds);
    }

    // Helper function to get the token ID from the vocab
    private static float getTokenId(String token, String[] vocab) {
        for (int i = 0; i < vocab.length; i++) {
            if (vocab[i].equals(token)) {
                return i;
            }
        }
        return -1; // Return a special token ID for unknown words
    }

    // Helper function to get the top-k words based on the output probabilities
    private static List<String> getTopKWords(Tensor<Float> output, String[] vocab, int beamWidth) {
        float[] probabilities = output.copyTo(new float[1][vocab.length])[0];
        List<String> topKWords = new ArrayList<>();
        
        // Create a list of words and their probabilities
        List<WordProbability> wordProbabilities = new ArrayList<>();
        for (int i = 0; i < vocab.length; i++) {
            wordProbabilities.add(new WordProbability(vocab[i], probabilities[i]));
        }

        // Sort by probability and take the top-k
        wordProbabilities.sort((a, b) -> Float.compare(b.probability, a.probability));
        for (int i = 0; i < beamWidth; i++) {
            topKWords.add(wordProbabilities.get(i).word);
        }

        return topKWords;
    }

    // Helper function to get the word probability from the output tensor
    private static float getWordProbability(String word, Tensor<Float> output) {
        // Get the index for the word and retrieve its probability
        int wordId = getTokenId(word, output.shape());
        float[] probabilities = output.copyTo(new float[1][output.shape()[1]])[0];
        return probabilities[wordId];
    }

    // Helper function to get the top-k sequences (sorted by score)
    private static List<BeamSearchNode> getTopKSequences(List<BeamSearchNode> allCandidates, int beamWidth) {
        allCandidates.sort((a, b) -> Float.compare(b.score, a.score));  // Sort by score
        return allCandidates.subList(0, beamWidth);  // Keep the top-k sequences
    }

    // BeamSearchNode class to keep track of the sequences and scores
    static class BeamSearchNode {
        String sequence;
        float score;
        Tensor<Float> imageFeatures;

        BeamSearchNode(String sequence, float score, Tensor<Float> imageFeatures) {
            this.sequence = sequence;
            this.score = score;
            this.imageFeatures = imageFeatures;
        }

        List<String> getSequenceTokens() {
            List<String> tokens = new ArrayList<>();
            for (String token : sequence.split(" ")) {
                tokens.add(token);
            }
            return tokens;
        }
    }

    // WordProbability class to store word and its probability for sorting
    static class WordProbability {
        String word;
        float probability;

        WordProbability(String word, float probability) {
            this.word = word;
            this.probability = probability;
        }
    }
}