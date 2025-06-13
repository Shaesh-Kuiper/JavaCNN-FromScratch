// CNNModel.java

public class CNNModel implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private Conv2D c1, c2, c3, c4;
    private ReLU r1, r2, r3, r4;
    private MaxPool2D p1, p2;
    private Flatten fl;
    private Dense fc;

    public CNNModel() {
        c1 = new Conv2D(1, 8, 3);
        r1 = new ReLU();
        c2 = new Conv2D(8, 8, 3);
        r2 = new ReLU();
        p1 = new MaxPool2D();

        c3 = new Conv2D(8, 16, 3);
        r3 = new ReLU();
        c4 = new Conv2D(16, 16, 3);
        r4 = new ReLU();
        p2 = new MaxPool2D();

        fl = new Flatten();
        // after two valid conv+pool: 28->26->24->12->10->8->4 => 16*4*4=256
        fc = new Dense(16*4*4, 10);
    }

    public float[] forward(float[][][] x) {
        float[][][] y = c1.forward(x);
        y = r1.forward(y);
        y = c2.forward(y);
        y = r2.forward(y);
        y = p1.forward(y);

        y = c3.forward(y);
        y = r3.forward(y);
        y = c4.forward(y);
        y = r4.forward(y);
        y = p2.forward(y);

        float[] v = fl.forward(y);
        return fc.forward(v);
    }

    public void backward(float[] grad, float lr) {
        float[] g1 = fc.backward(grad, lr);
        float[][][] g2 = fl.backward(g1);

        g2 = p2.backward(g2);
        g2 = r4.backward(g2);
        g2 = c4.backward(g2, lr);
        g2 = r3.backward(g2);
        g2 = c3.backward(g2, lr);

        g2 = p1.backward(g2);
        g2 = r2.backward(g2);
        g2 = c2.backward(g2, lr);
        g2 = r1.backward(g2);
        c1.backward(g2, lr);
    }

    public int predict(float[][][] x) {
        float[] logits = forward(x);
        int best=0; float m=logits[0];
        for(int i=1;i<logits.length;i++){
            if(logits[i]>m){ m=logits[i]; best=i; }
        }
        return best;
    }
}
