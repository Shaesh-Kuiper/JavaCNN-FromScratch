// Utils.java

import java.io.*;

public class Utils {
    public static class DataSet {
        public float[][][][] images; // [N][1][28][28]
        public int[] labels;
        public DataSet(float[][][][] imgs, int[] labs) {
            images=imgs; labels=labs;
        }
    }

    public static DataSet loadData(String imgPath, String lblPath) throws IOException {
        DataInputStream imgIn = new DataInputStream(new FileInputStream(imgPath));
        int magic = imgIn.readInt();
        int N = imgIn.readInt();
        int H = imgIn.readInt();
        int W = imgIn.readInt();
        float[][][][] images = new float[N][1][H][W];
        for(int i=0;i<N;i++){
            for(int r=0;r<H;r++){
                for(int c=0;c<W;c++){
                    int pix = imgIn.readUnsignedByte();
                    images[i][0][r][c] = pix/255.f;
                }
            }
        }
        imgIn.close();

        DataInputStream lblIn = new DataInputStream(new FileInputStream(lblPath));
        lblIn.readInt();
        int M = lblIn.readInt();
        int[] labels = new int[M];
        for(int i=0;i<M;i++) labels[i] = lblIn.readUnsignedByte();
        lblIn.close();

        return new DataSet(images, labels);
    }
}
