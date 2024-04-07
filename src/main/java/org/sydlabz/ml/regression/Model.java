package org.sydlabz.ml.regression;

import static org.sydlabz.ml.regression.ArrayOps.*;

public class Model {
    public static double mse(double[] yValues, double[] yPredictions) {
        double[] squaredDifferences = square(diff(yValues, yPredictions));

        return sum(squaredDifferences) / yValues.length;
    }

    public static double r2(double[] yValues, double[] yPredictions) {
        double meanOfYValues = mean(yValues);

        double[] yDispersions = dispersion(yValues, meanOfYValues);

        double[] yVariability = variability(yPredictions, meanOfYValues);

        double residualSumOfSquares = sum(square(yVariability));
        double totalSumOfSquares = sum(square(yDispersions));

        return residualSumOfSquares / totalSumOfSquares;
    }

    public static double see(double[] yValues, double[] yPredictions) {
        double[] squaredDifferences = square(diff(yPredictions, yValues));

        return Math.sqrt(sum(squaredDifferences) / (yValues.length - 2));
    }

}
