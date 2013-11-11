using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class ActivationFunction
    {
        public Func<double, double> Function
        {
            get;
            set;
        }

        public Func<double, double> Derivative
        {
            get;
            set;
        }
    }

    public static class ActivationFunctions
    {
        public static ActivationFunction Sigmoid = new ActivationFunction()
        {
            Function = (double x) =>
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            },
            Derivative = (double x) =>
            {
                return x * (1.0 - x);
            }
        };

        public static ActivationFunction Tanh = new ActivationFunction()
        {
            Function = (double x) =>
            {
                return Math.Tanh(x);
            },
            Derivative = (double x) =>
            {
                return (1.0 + x) * (1.0 - x);
            }
        };
    }
}
