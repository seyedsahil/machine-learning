package org.sydlabz.ml.regression;

import static java.util.stream.IntStream.range;
import static org.sydlabz.ml.regression.ArrayOps.*;

public abstract class DualLinearModel extends Model {
    public final double commonIntercept;
    public final double slopeForX1;
    public final double slopeForX2;

    public DualLinearModel(double commonIntercept, double slopeForX1, double slopeForX2) {
        this.commonIntercept = commonIntercept;
        this.slopeForX1 = slopeForX1;
        this.slopeForX2 = slopeForX2;
    }

    public static DualLinearModel train(double[] x1Values, double[] x2Values, double[] yValues) {
        double[][] xValues = {x1Values, x2Values};

        double sumOfYValues = sum(yValues);
        double[] sumOfXValues = sum(xValues);

        double meanOfYValues = mean(yValues);
        double[] meanOfXValues = mean(xValues);

        double[] sumOfSquaresOfXValues = sum(square(xValues));
        double[] sumOfProductsOfXValuesYValues = sum(product(xValues, yValues));
        double sumOfProductsOfXValues = sum(product(xValues));

        int totalFeatures = xValues.length;
        int totalObservations = yValues.length;

        double[] rSumSquaresOfXValues = range(0, totalFeatures).mapToDouble(index -> rSum(totalObservations, sumOfSquaresOfXValues[index], sumOfXValues[index], sumOfXValues[index])).toArray();
        double[] rSumProductsOfXValuesAndYValues = range(0, totalFeatures).mapToDouble(index -> rSum(totalObservations, sumOfProductsOfXValuesYValues[index], sumOfXValues[index], sumOfYValues)).toArray();
        double rSumProductsOfXValues = rSum(totalObservations, sumOfProductsOfXValues, sumOfXValues[0], sumOfXValues[1]);

        double denominator = rSumSquaresOfXValues[0] * rSumSquaresOfXValues[1] - rSumProductsOfXValues * rSumProductsOfXValues;

        double slopeForX1 = (rSumSquaresOfXValues[1] * rSumProductsOfXValuesAndYValues[0] - rSumProductsOfXValues * rSumProductsOfXValuesAndYValues[1]) / denominator;
        double slopeForX2 = (rSumSquaresOfXValues[0] * rSumProductsOfXValuesAndYValues[1] - rSumProductsOfXValues * rSumProductsOfXValuesAndYValues[0]) / denominator;

        double commonIntercept = meanOfYValues - slopeForX1 * meanOfXValues[0] - slopeForX2 * meanOfXValues[1];

        return new DualLinearModel(commonIntercept, slopeForX1, slopeForX2) {
            @Override
            public double predict(double value1, double value2) {
                return this.commonIntercept + this.slopeForX1 * value1 + this.slopeForX2 * value2;
            }
        };
    }

    private static double rSum(int size, double value1, double value2, double value3) {
        return value1 - value2 * value3 / size;
    }

    abstract public double predict(double value1, double value2);

    public final double[] predict(double[] x1Values, double[] x2Values) {
        return range(0, x1Values.length).mapToDouble(index -> predict(x1Values[index], x2Values[index])).toArray();
    }

    @Override
    public String toString() {
        return "(commonIntercept=" + commonIntercept + ", slopeForX1=" + slopeForX1 + ", slopeForX2=" + slopeForX2 + ")";
    }
}
