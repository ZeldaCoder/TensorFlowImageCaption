import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.GraphDef;

public class ImageEncoder {

    public static Tensor<Float> encodeImage(String imagePath) {
        // Load a pre-trained CNN model (e.g., InceptionV3 or ResNet)
        // You would load the model into a Graph here
        // Assuming the graph is already set up for feature extraction
        
        Graph graph = new Graph();
        byte[] graphBytes = Files.readAllBytes(Paths.get("path/to/inception_v3_model.pb"));
        graph.importGraphDef(graphBytes);

        try (Session session = new Session(graph)) {
            // Preprocess the image (resize, normalize, etc.)
            // Create a tensor for the image to feed into the model
            Tensor<Float> imageTensor = preprocessImage(imagePath);

            // Run the model and extract the features
            Tensor<Float> features = session.runner()
                    .feed("input_tensor_name", imageTensor)
                    .fetch("output_tensor_name")
                    .run().get(0);

            return features;
        }
    }

    private static Tensor<Float> preprocessImage(String imagePath) {
       // Load the image from file
        BufferedImage image = ImageIO.read(new File(imagePath));

        // Resize the image to the required dimensions (299x299 for InceptionV3, for example)
        int targetWidth = 299;
        int targetHeight = 299;
        image = resizeImage(image, targetWidth, targetHeight);

        // Convert the image to float and normalize it
        float[] imageArray = new float[targetWidth * targetHeight * 3]; // 3 channels (RGB)
        int idx = 0;
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                // Get RGB value at pixel (x, y)
                Color color = new Color(image.getRGB(x, y));
                // Normalize the RGB values between 0 and 1
                imageArray[idx++] = color.getRed() / 255.0f; // Red channel
                imageArray[idx++] = color.getGreen() / 255.0f; // Green channel
                imageArray[idx++] = color.getBlue() / 255.0f; // Blue channel
            }
        }

        // Convert the image to a Tensor
        Tensor<Float> imageTensor = Tensor.create(new long[]{1, targetHeight, targetWidth, 3}, imageArray);
        return imageTensor;
    }

     // Resize the image to the target width and height
     private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        Image scaledImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(scaledImage, 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }

    // Subtract mean pixel values for InceptionV3 or any other model's required mean
private static float[] normalizeImage(float[] imageArray) {
    float[] normalizedImage = new float[imageArray.length];
    // Mean values for InceptionV3 (R, G, B)
    float[] mean = {0.485f, 0.456f, 0.406f};
    
    int idx = 0;
    for (int i = 0; i < imageArray.length; i++) {
        int channel = i % 3;  // R, G, B
        normalizedImage[i] = imageArray[i] - mean[channel];
        idx++;
    }
    return normalizedImage;
}

}

