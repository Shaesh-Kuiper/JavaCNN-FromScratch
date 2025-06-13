// Dense.java

import java.util.*;

public class Dense implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private int inFeatures, outFeatures;
    private float[][] W;  // [out][in]
    private float[] b;    // [out]
    private float[] input; 

    public Dense(int inF, int outF) {
        inFeatures = inF; outFeatures = outF;
        W = new float[outF][inF];
        b = new float[outF];
        Random rnd = new Random();
        float std = (float)Math.sqrt(2.0/(inF+outF));
        for(int i=0;i<outF;i++){
            b[i]=0;
            for(int j=0;j<inF;j++)
                W[i][j] = (float)(rnd.nextGaussian()*std);
        }
    }

    public float[] forward(float[] x) {
        input = x;
        float[] out = new float[outFeatures];
        for(int i=0;i<outFeatures;i++){
            float sum = b[i];
            for(int j=0;j<inFeatures;j++)
                sum += W[i][j] * x[j];
            out[i]=sum;
        }
        return out;
    }

    public float[] backward(float[] gradOut, float lr) {
        float[] gradIn = new float[inFeatures];
        // weight & input grads
        for(int i=0;i<outFeatures;i++){
            float g = gradOut[i];
            b[i] -= lr * g;
            for(int j=0;j<inFeatures;j++){
                gradIn[j] += W[i][j] * g;
                W[i][j] -= lr * g * input[j];
            }
        }
        return gradIn;
    }
}
