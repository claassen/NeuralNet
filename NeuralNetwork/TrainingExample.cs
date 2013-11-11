using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class TrainingExample
    {
        public double[] Input
        {
            get;
            set;
        }

        public double[] Expected
        {
            get;
            set;
        }

        public TrainingExample(double[] input, double[] expected)
        {
            Input = input;
            Expected = expected;
        }
    }
}
