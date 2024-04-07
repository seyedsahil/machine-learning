package org.sydlabz.ml.regression;

import java.util.stream.Collectors;

import static java.util.Arrays.stream;
import static java.util.stream.IntStream.range;

/**
 * @author seyedsahil
 */
public final class ArrayOps {
    public static void usable(double[]... values) {
        stream(values).forEach(value -> {
            if (value == null || value.length == 0) {
                throw new IllegalArgumentException("Input Error: null or zero length");
            }
        });
    }

    private static void match(double[]... values) {
        long threshold = stream(values).mapToInt(value -> value.length).distinct().count();

        if (threshold != 1) {
            throw new IllegalArgumentException("Input Error: length not matching");
        }
    }

    public static double sum(double[] values) {
        usable(values);

        return stream(values).sum();
    }

    public static double[] sum(double[][] values) {
        usable(values);

        return stream(values).mapToDouble(ArrayOps::sum).toArray();
    }

    public static double mean(double[] values) {
        usable(values);

        return stream(values).average().orElse(Double.NaN);
    }

    public static double[] mean(double[][] values) {
        usable(values);

        return stream(values).mapToDouble(ArrayOps::mean).toArray();
    }

    public static double[] diff(double[] values1, double[] values2) {
        usable(values1, values2);
        match(values1, values2);

        return range(0, values1.length).mapToDouble(index -> values1[index] - values2[index]).toArray();
    }

    public static double[] diff(double[] values, double constant) {
        usable(values);

        return stream(values).map(value -> value - constant).toArray();
    }

    public static double[] product(double[] values1, double[] values2) {
        usable(values1, values2);
        match(values1, values2);

        return range(0, values1.length).mapToDouble(index -> values1[index] * values2[index]).toArray();
    }

    public static double[][] product(double[][] values1, double[] values2) {
        usable(values1);
        usable(values2);
        match(values1);
        match(values1[0], values2);

        return stream(values1).map(values -> product(values, values2)).toArray(double[][]::new);
    }

    public static double[] product(double[][] values) {
        usable(values);
        match(values);

        return stream(values).reduce((accumulated, row) -> range(0, row.length).mapToDouble(index -> accumulated[index] * row[index]).toArray()).orElse(new double[0]);
    }

    public static double[] square(double[] values) {
        usable(values);

        return stream(values).map(value -> value * value).toArray();
    }

    public static double[][] square(double[][] values) {
        usable(values);

        return stream(values).map(ArrayOps::square).toArray(double[][]::new);
    }

    public static String string(double[] values) {
        usable(values);

        return stream(values).mapToObj(String::valueOf).collect(Collectors.joining(", "));
    }

    public static String string(double[][] values) {
        usable(values);

        return stream(values).map(ArrayOps::string).collect(Collectors.joining("\n"));
    }

    public static double covariance(double[] xDispersions, double[] yDispersions) {
        return sum(product(xDispersions, yDispersions));
    }

    public static double variance(double[] xDispersions) {
        return sum(square(xDispersions));
    }

    public static double[] dispersion(double[] values, double mean) {
        return diff(values, mean);
    }

    public static double[] variability(double[] values, double mean) {
        return diff(values, mean);
    }
}
