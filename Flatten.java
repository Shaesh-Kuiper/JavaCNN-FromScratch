// Flatten.java

public class Flatten implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private int inC, inH, inW;

    public float[] forward(float[][][] x) {
        inC=x.length; inH=x[0].length; inW=x[0][0].length;
        float[] out = new float[inC*inH*inW];
        int idx=0;
        for(int c=0;c<inC;c++)
            for(int i=0;i<inH;i++)
                for(int j=0;j<inW;j++)
                    out[idx++]=x[c][i][j];
        return out;
    }

    public float[][][] backward(float[] grad) {
        float[][][] out = new float[inC][inH][inW];
        int idx=0;
        for(int c=0;c<inC;c++)
            for(int i=0;i<inH;i++)
                for(int j=0;j<inW;j++)
                    out[c][i][j]=grad[idx++];
        return out;
    }
}
