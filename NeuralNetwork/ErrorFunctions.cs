using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    /*public enum ErrorFunctionType
    {
        SquareError = 1,
        MeanSquareError = 2,
        CrossEntropy = 3
    }

    public class ErrorFunction
    {
        public Func<double, double, double> Derivative
        {
            get;
            set;
        }
    }

    public static class ErrorFunctions
    {
        public static ErrorFunction Get(ErrorFunctionType type)
        {
            switch (type)
            {
                case ErrorFunctionType.SquareError:
                case ErrorFunctionType.MeanSquareError:
                case ErrorFunctionType.CrossEntropy:
            }
        }

        public static ErrorFunction SquareError = new ErrorFunction()
        {
            Derivative = (double expected, double actual) =>
            {
                return expected - actual;
            }
        };
    }*/
}
