import java.io.IOException;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;
import java.awt.BorderLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;

public class Inference {
    // Map label indices to class names
    private static final String[] CLASS_NAMES = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length < 4) {
            System.err.println("Usage: java Inference <modelPath> <imageFile> <labelFile> <index>");
            System.exit(1);
        }
        String modelPath = args[0];
        String imgPath   = args[1];
        String lblPath   = args[2];
        int idx          = Integer.parseInt(args[3]);

        // 1. Load the model
        CNNModel model = ModelIO.loadModel(modelPath);

        // 2. Load the dataset and pick one example
        Utils.DataSet ds = Utils.loadData(imgPath, lblPath);
        float[][][] img = ds.images[idx];
        int trueLabel = ds.labels[idx];

        // 3. Show the image
        showImage(img, 100, 100);

        // 4. Forward pass to get logits
        float[] logits = model.forward(img);

        // 5. Softmax → probabilities
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > maxLogit) maxLogit = v;
        float sumExp = 0f;
        for (float v : logits) sumExp += Math.exp(v - maxLogit);
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float)(Math.exp(logits[i] - maxLogit) / sumExp);
        }

        // 6. Find prediction and confidence
        int pred = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[pred]) pred = i;
        }
        float confidence = probs[pred] * 100f;

        // 7. Print results with class names
        System.out.printf(
            "Predicted: %d (%s) — %.2f%% confidence%n",
            pred, CLASS_NAMES[pred], confidence
        );
        System.out.printf(
            "True label: %d (%s)%n",
            trueLabel, CLASS_NAMES[trueLabel]
        );
    }

    /** 
     * Converts a [C=1][H][W] float image (values in [0,1]) into a grayscale BufferedImage
     * and displays it at the given on-screen position.
     */
    private static void showImage(float[][][] img, int x, int y) {
        int H = img[0].length, W = img[0][0].length;
        BufferedImage bimg = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int v = Math.min(255, Math.max(0, (int)(img[0][i][j] * 255)));
                int rgb = (v << 16) | (v << 8) | v;
                bimg.setRGB(j, i, rgb);
            }
        }

        // Nearest‐neighbor upscaling to 280×280
        Image scaled = bimg.getScaledInstance(560, 560, Image.SCALE_REPLICATE);

        JFrame frame = new JFrame("Input Image");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        
        frame.setLayout(new BorderLayout());
        frame.add(new JLabel(new ImageIcon(scaled)), BorderLayout.CENTER);
        frame.pack();
        frame.setLocation(x, y);
        frame.setVisible(true);
    }
}
