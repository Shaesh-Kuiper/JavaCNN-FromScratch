// ThreadPool.java

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadPool {
    // fix at 4 threads (50% of 8 cores)
    public static final int NUM_THREADS = 4;
    public static final ExecutorService POOL = Executors.newFixedThreadPool(NUM_THREADS);
}
