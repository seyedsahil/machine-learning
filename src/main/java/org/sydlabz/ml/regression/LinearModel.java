package org.sydlabz.ml.regression;

import static java.util.Arrays.stream;
import static org.sydlabz.ml.regression.ArrayOps.*;

public abstract class LinearModel {
    public static LinearModel train(double[] xValues, double[] yValues) {
        double meanOfXValues = mean(xValues);
        double meanOfYValues = mean(yValues);

        double[] xDispersions = dispersion(xValues, meanOfXValues);
        double[] yDispersions = dispersion(yValues, meanOfYValues);

        double variance = variance(xDispersions);
        double covariance = covariance(xDispersions, yDispersions);

        double slopeOfLine = covariance / variance;
        double yIntercept = meanOfYValues - slopeOfLine * meanOfXValues;

        return new LinearModel() {
            @Override
            public double predict(double xValue) {
                return yIntercept + slopeOfLine * xValue;
            }
        };
    }

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

    abstract public double predict(double value);

    public final double[] predict(double[] values) {
        return stream(values).map(this::predict).toArray();
    }
}
