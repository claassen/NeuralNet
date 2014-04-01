using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Functions;

namespace NeuralNetwork.Layers
{
    public class HiddenLayer : Layer
    {
        public HiddenLayer() { }

        public HiddenLayer(ActivationFunctionType activationFunctionType, int numNodes)
            : base(activationFunctionType, numNodes)
        {
        }

        public override double Backpropagate(LearningMethod learningMethod, Layer previous, Layer next, double learningRate, double momentum, double weightDecay, MiniBatchMode miniBatchMode)
        {
            if (miniBatchMode == MiniBatchMode.Off || miniBatchMode == MiniBatchMode.First)
            {
                ResetWeightGradients();
            }

            /* Backpropagation:
             * 
             *   dE/dO(i) = activation_derivative(O(j)) * sum[w(i->j) * dE/dO(j)] for all nodes in layer j
             *  
             * i = this layer
             * j = previous (higher) layer
             */
            for (int i = 0; i < NumNodes; i++)
            {
                double outputGradientSum = 0;
                for (int k = 0; k < previous.OutputGradients.Length; k++)
                {
                    outputGradientSum += previous.OutputGradients[k] * previous.Weights[k * previous.NumWeightsPerNode + i]; 
                }

                OutputGradients[i] = ActivationFunctions.Get(ActivationFuncType).Derivative(Outputs[i]) * outputGradientSum;
            }

            UpdateWeightGradients(next.Outputs);
            UpdateWeights(learningMethod, miniBatchMode, learningRate, momentum, weightDecay);

            return 0;
        }
    }
}
