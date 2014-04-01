using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Functions;
using NeuralNetwork.DataSetProviders;

namespace NeuralNetwork.Layers
{
    public class InputLayer : Layer
    {
        public IDataSetProvider DataSetProvider;

        public InputLayer() { }

        public InputLayer(IDataSetProvider dataProvider)
            : base(ActivationFunctionType.Sigmoid, dataProvider.InputSize())
        {
            Init(0);
            DataSetProvider = dataProvider;
        }

        public override void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = 1;
            NumFeatureMaps = 1;
        }

        public override double[] Evaluate(double[] input)
        {
            if (DataSetProvider.ScaleInput())
            {
                for (int i = 0; i < input.Length; i++)
                {
                    Outputs[i] = ScaleValue(input[i]);
                }
            }
            else
            {
                Outputs = input;
            }

            return Outputs;
        }

        private double ScaleValue(double value)
        {
            return (((value - DataSetProvider.InputMin()) * (DataSetProvider.ScaleMax() - DataSetProvider.ScaleMin())) / (DataSetProvider.InputMax() - DataSetProvider.InputMin())) + DataSetProvider.ScaleMin();   
        }
    }
}
