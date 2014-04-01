using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.Functions
{
    public enum ActivationFunctionType
    {
        Sigmoid = 1,
        Tanh = 2,
        Softmax = 3
    }

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
        public static ActivationFunction Get(ActivationFunctionType type)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid: return ActivationFunctions.Sigmoid;
                case ActivationFunctionType.Tanh:    return ActivationFunctions.Tanh;
                case ActivationFunctionType.Softmax: return ActivationFunctions.Softmax;
                default: return null;
            }
        }

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

        public static ActivationFunction Softmax = new ActivationFunction()
        {
            Function = (double x) =>
            {
                throw new NotImplementedException("Softmax needs to be handled specially, it does not have a function implementation");
            },
            Derivative = (double x) =>
            {
                throw new NotImplementedException("Softmax needs to be handled specially, it does not have a function implementation");
            }
        };
    }
}
