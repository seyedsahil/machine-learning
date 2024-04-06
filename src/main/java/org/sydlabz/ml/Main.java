package org.sydlabz.ml;

import org.sydlabz.ml.regression.LinearModel;

import static org.sydlabz.ml.regression.ArrayOps.string;

public class Main {
    public static void main(String[] args) {
        test1();
    }

    private static void test1() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 5, 4, 5};

        LinearModel linearModel = LinearModel.train(x, y);

        double[] yPredictions = linearModel.predict(x);

        double r2 = LinearModel.r2(y, yPredictions);
        double mse = LinearModel.mse(y, yPredictions);
        double see = LinearModel.see(y, yPredictions);

        System.out.println("Actual:         " + string(y));
        System.out.println("Predictions:    " + string(yPredictions));
        System.out.println("MSE:            " + mse);
        System.out.println("R2:             " + r2);
        System.out.println("SEE:            " + see);
    }
}
