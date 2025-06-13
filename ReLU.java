// ReLU.java

public class ReLU implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private float[][][] mask3;
    private float[] mask1;

    public float[][][] forward(float[][][] x) {
        int C=x.length, H=x[0].length, W=x[0][0].length;
        float[][][] out = new float[C][H][W];
        mask3 = new float[C][H][W];
        for(int c=0;c<C;c++){
            for(int i=0;i<H;i++){
                for(int j=0;j<W;j++){
                    if(x[c][i][j]>0) { out[c][i][j]=x[c][i][j]; mask3[c][i][j]=1; }
                    else { out[c][i][j]=0; mask3[c][i][j]=0; }
                }
            }
        }
        return out;
    }

    public float[][][] backward(float[][][] grad) {
        int C=grad.length, H=grad[0].length, W=grad[0][0].length;
        float[][][] out = new float[C][H][W];
        for(int c=0;c<C;c++)
            for(int i=0;i<H;i++)
                for(int j=0;j<W;j++)
                    out[c][i][j] = grad[c][i][j] * mask3[c][i][j];
        return out;
    }
}
