package org.sydlabz.ml.regression;

import static org.sydlabz.ml.regression.ArrayOps.string;
import static org.sydlabz.ml.regression.Model.*;

public class Tests {
    public static void test1() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 5, 4, 5};

        LinearModel linearModel = LinearModel.train(x, y);

        double[] yPredictions = linearModel.predict(x);

        double r2 = r2(y, yPredictions);
        double mse = mse(y, yPredictions);
        double see = see(y, yPredictions);

        System.out.println("Model:          " + linearModel);
        System.out.println("Actual:         " + string(y));
        System.out.println("Predictions:    " + string(yPredictions));
        System.out.println("MSE:            " + mse);
        System.out.println("R2:             " + r2);
        System.out.println("SEE:            " + see);
    }

    public static void test2() {
        double[] x1Values = {60, 62, 67, 70, 71, 72, 75, 78};
        double[] x2Values = {22, 25, 24, 20, 15, 14, 14, 11};
        double[] yValues = {140, 155, 159, 179, 192, 200, 212, 215};

        DualLinearModel dualLinearModel = DualLinearModel.train(x1Values, x2Values, yValues);

        double[] yPredictions = dualLinearModel.predict(x1Values, x2Values);

        double r2 = r2(yValues, yPredictions);
        double mse = mse(yValues, yPredictions);
        double see = see(yValues, yPredictions);

        System.out.println("Model:          " + dualLinearModel);
        System.out.println("Actual:         " + string(yValues));
        System.out.println("Predictions:    " + string(yPredictions));
        System.out.println("MSE:            " + mse);
        System.out.println("R2:             " + r2);
        System.out.println("SEE:            " + see);
    }
}
