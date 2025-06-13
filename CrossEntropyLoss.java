// CrossEntropyLoss.java

public class CrossEntropyLoss {
    // returns loss for one sample
    public static float forward(float[] logits, int label) {
        float max = Float.NEGATIVE_INFINITY;
        for(float v: logits) if(v>max) max=v;
        float sum=0;
        for(float v: logits) sum += Math.exp(v-max);
        float logProb = logits[label]-max - (float)Math.log(sum);
        return -logProb;
    }

    // returns dL/dlogits
    public static float[] backward(float[] logits, int label) {
        int n = logits.length;
        float max = Float.NEGATIVE_INFINITY;
        for(float v: logits) if(v>max) max=v;
        float sum=0;
        for(float v: logits) sum += Math.exp(v-max);
        float[] grad = new float[n];
        for(int i=0;i<n;i++){
            float p = (float)Math.exp(logits[i]-max)/sum;
            grad[i] = p - (i==label ? 1f : 0f);
        }
        return grad;
    }
}
