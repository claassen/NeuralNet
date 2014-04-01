using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Functions;

namespace NeuralNetwork.Layers
{
    public class FullyConnectedLayer : HiddenLayer
    {
        public FullyConnectedLayer() { }

        public FullyConnectedLayer(ActivationFunctionType activationFunctionType, int numNodes)
            : base(activationFunctionType, numNodes)
        {
        }
    }
}
