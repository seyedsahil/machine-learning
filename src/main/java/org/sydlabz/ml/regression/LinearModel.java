package org.sydlabz.ml.regression;

import static java.util.Arrays.stream;
import static org.sydlabz.ml.regression.ArrayOps.*;

public abstract class LinearModel extends Model {
    public final double yIntercept;
    public final double slopeOfLine;

    public LinearModel(double yIntercept, double slopeOfLine) {
        this.yIntercept = yIntercept;
        this.slopeOfLine = slopeOfLine;
    }

    public static LinearModel train(double[] xValues, double[] yValues) {
        double meanOfXValues = mean(xValues);
        double meanOfYValues = mean(yValues);

        double[] xDispersions = dispersion(xValues, meanOfXValues);
        double[] yDispersions = dispersion(yValues, meanOfYValues);

        double variance = variance(xDispersions);
        double covariance = covariance(xDispersions, yDispersions);

        double slopeOfLine = covariance / variance;
        double yIntercept = meanOfYValues - slopeOfLine * meanOfXValues;

        return new LinearModel(yIntercept, slopeOfLine) {
            @Override
            public double predict(double xValue) {
                return this.yIntercept + this.slopeOfLine * xValue;
            }
        };
    }

    abstract public double predict(double value);

    public final double[] predict(double[] values) {
        return stream(values).map(this::predict).toArray();
    }

    @Override
    public String toString() {
        return "(yIntercept=" + yIntercept + ", slopeOfLine=" + slopeOfLine + ")";
    }
}
