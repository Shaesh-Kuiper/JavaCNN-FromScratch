// Conv2D.java

import java.util.*;
import java.util.concurrent.*;

public class Conv2D implements java.io.Serializable{
    private static final long serialVersionUID = 1L;
    private int inChannels, outChannels, kernelSize;
    private float[][][][] weights;   // [outC][inC][k][k]
    private float[] bias;            // [outC]
    private float[][][] input;       // saved for backward
    private float[][][][] gradW;
    private float[] gradB;

    public Conv2D(int inChannels, int outChannels, int kernelSize) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        weights = new float[outChannels][inChannels][kernelSize][kernelSize];
        bias = new float[outChannels];
        gradW = new float[outChannels][inChannels][kernelSize][kernelSize];
        gradB = new float[outChannels];
        // Xavier init
        Random rnd = new Random();
        float std = (float)Math.sqrt(2.0/(inChannels*kernelSize*kernelSize+outChannels));
        for(int f=0;f<outChannels;f++){
            bias[f]=0;
            for(int c=0;c<inChannels;c++){
                for(int i=0;i<kernelSize;i++){
                    for(int j=0;j<kernelSize;j++){
                        weights[f][c][i][j] = (float)(rnd.nextGaussian()*std);
                    }
                }
            }
        }
    }

    public float[][][] forward(float[][][] x) {
        input = x;
        int H = x[0].length, W = x[0][0].length;
        int outH = H - kernelSize + 1, outW = W - kernelSize + 1;
        float[][][] out = new float[outChannels][outH][outW];

        List<Callable<Void>> tasks = new ArrayList<>();
        int chunk = (outChannels + ThreadPool.NUM_THREADS -1)/ThreadPool.NUM_THREADS;
        for(int t=0; t<ThreadPool.NUM_THREADS; t++){
            final int start = t*chunk, end = Math.min(start+chunk, outChannels);
            if(start>=end) continue;
            tasks.add(() -> {
                for(int f=start; f<end; f++){
                    for(int i=0; i<outH; i++){
                        for(int j=0; j<outW; j++){
                            float sum = bias[f];
                            for(int c=0; c<inChannels; c++){
                                for(int ki=0; ki<kernelSize; ki++){
                                    for(int kj=0; kj<kernelSize; kj++){
                                        sum += weights[f][c][ki][kj] * x[c][i+ki][j+kj];
                                    }
                                }
                            }
                            out[f][i][j] = sum;
                        }
                    }
                }
                return null;
            });
        }
        try {
            ThreadPool.POOL.invokeAll(tasks);
        } catch(InterruptedException e) {
            throw new RuntimeException(e);
        }
        return out;
    }

    public float[][][] backward(float[][][] gradOut, float lr) {
        int H = input[0].length, W = input[0][0].length;
        int outH = gradOut[0].length, outW = gradOut[0][0].length;
        // zero grads
        for(int f=0;f<outChannels;f++){
            gradB[f]=0;
            for(int c=0;c<inChannels;c++)
                for(int i=0;i<kernelSize;i++)
                    Arrays.fill(gradW[f][c][i], 0);
        }
        float[][][] gradIn = new float[inChannels][H][W];

        // compute weight & bias grads, and grad w.r.t input (sequential to avoid races)
        for(int f=0; f<outChannels; f++){
            for(int i=0; i<outH; i++){
                for(int j=0; j<outW; j++){
                    float g = gradOut[f][i][j];
                    gradB[f] += g;
                    for(int c=0; c<inChannels; c++){
                        for(int ki=0; ki<kernelSize; ki++){
                            for(int kj=0; kj<kernelSize; kj++){
                                gradW[f][c][ki][kj] += g * input[c][i+ki][j+kj];
                                gradIn[c][i+ki][j+kj] += g * weights[f][c][ki][kj];
                            }
                        }
                    }
                }
            }
        }

        // update
        for(int f=0;f<outChannels;f++){
            bias[f] -= lr * gradB[f];
            for(int c=0;c<inChannels;c++){
                for(int i=0;i<kernelSize;i++){
                    for(int j=0;j<kernelSize;j++){
                        weights[f][c][i][j] -= lr * gradW[f][c][i][j];
                    }
                }
            }
        }
        return gradIn;
    }
}
