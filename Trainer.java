// Trainer.java

import java.io.IOException;
import java.util.*;

public class Trainer {
    public static void main(String[] args) throws IOException {
        // load
        Utils.DataSet train = Utils.loadData("C:\\learning\\P\\data\\FashionMNIST\\raw\\train-images-idx3-ubyte", "C:\\learning\\P\\data\\FashionMNIST\\raw\\train-labels-idx1-ubyte");
        Utils.DataSet test  = Utils.loadData("C:\\learning\\P\\data\\FashionMNIST\\raw\\t10k-images-idx3-ubyte", "C:\\learning\\P\\data\\FashionMNIST\\raw\\t10k-labels-idx1-ubyte");

        System.out.println("Loaded the dataset ");
        System.out.println();

        CNNModel model = new CNNModel();

        System.out.println("model Initiated");
        System.out.println();

        int epochs = 2, batchSize = 64;
        float lr = 0.01f;
        int N = train.images.length;
        Random rnd = new Random();

        System.out.println("Training started");

        for(int ep=1; ep<=epochs; ep++){
            // shuffle
            System.out.println("Epoch : " + ep);
            System.out.println();
            Integer[] idx = new Integer[N];
            for(int i=0;i<N;i++) idx[i]=i;
            Collections.shuffle(Arrays.asList(idx), rnd);
            

            float totalLoss = 0;
            for(int b=0; b<N; b+=batchSize){
                int end = Math.min(b+batchSize, N);
                for(int i=b;i<end;i++){
                    int id = idx[i];
                    float[][][] x = train.images[id];
                    int lbl = train.labels[id];
                    float[] logits = model.forward(x);
                    float loss = CrossEntropyLoss.forward(logits, lbl);
                    totalLoss += loss;
                    float[] grad = CrossEntropyLoss.backward(logits, lbl);
                    model.backward(grad, lr);
                }
            }

            System.out.println("now evaluation");
            System.out.println();
            // eval
            int correct =0;
            for(int i=0;i<test.images.length;i++){
                if(model.predict(test.images[i])==test.labels[i]) correct++;
            }
            float acc = 100f*correct/test.images.length;
            System.out.printf("Epoch %d: avg loss=%.4f, test acc=%.2f%%%n",
                              ep, totalLoss/N, acc);
        }

        System.out.println("Training complete. Saving model...");
        try {
            ModelIO.saveModel(model, "trained-model.bin");
            System.out.println("Model saved to trained-model.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }

        ThreadPool.POOL.shutdown();
    }
}
