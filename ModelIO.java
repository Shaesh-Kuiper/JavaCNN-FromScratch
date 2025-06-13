import java.io.*;

public class ModelIO {
    public static void saveModel(CNNModel model, String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(model);
        }
    }

    public static CNNModel loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (CNNModel) ois.readObject();
        }
    }
}
