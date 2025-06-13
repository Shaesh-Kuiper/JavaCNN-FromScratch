// MaxPool2D.java

public class MaxPool2D implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private int poolSize=2;
    private int[][][] maxI, maxJ; // record indices

    public float[][][] forward(float[][][] x) {
        int C=x.length, H=x[0].length, W=x[0][0].length;
        int outH=H/poolSize, outW=W/poolSize;
        float[][][] out = new float[C][outH][outW];
        maxI = new int[C][outH][outW];
        maxJ = new int[C][outH][outW];
        for(int c=0;c<C;c++){
            for(int i=0;i<outH;i++){
                for(int j=0;j<outW;j++){
                    float m = Float.NEGATIVE_INFINITY;
                    for(int di=0;di<poolSize;di++){
                        for(int dj=0;dj<poolSize;dj++){
                            int ii=i*poolSize+di, jj=j*poolSize+dj;
                            if(x[c][ii][jj]>m){
                                m = x[c][ii][jj];
                                maxI[c][i][j]=ii;
                                maxJ[c][i][j]=jj;
                            }
                        }
                    }
                    out[c][i][j]=m;
                }
            }
        }
        return out;
    }

    public float[][][] backward(float[][][] gradOut) {
        int C=gradOut.length, outH=gradOut[0].length, outW=gradOut[0][0].length;
        int H=outH*poolSize, W=outW*poolSize;
        float[][][] gradIn = new float[C][H][W];
        for(int c=0;c<C;c++){
            for(int i=0;i<outH;i++){
                for(int j=0;j<outW;j++){
                    gradIn[c][ maxI[c][i][j] ][ maxJ[c][i][j] ] = gradOut[c][i][j];
                }
            }
        }
        return gradIn;
    }
}
