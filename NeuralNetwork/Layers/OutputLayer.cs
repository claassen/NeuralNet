using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Functions;

namespace NeuralNetwork.Layers
{
    public class OutputLayer : Layer
    {
        public OutputLayer() { }

        public OutputLayer(ActivationFunctionType activationFunctionType, int numNodes)
            : base(activationFunctionType, numNodes)
        {
        }

        public override double[] Evaluate(double[] input)
        {
            if (ActivationFuncType == ActivationFunctionType.Softmax)
            {
                //Uses the trick described here: http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
                //for handling values where exp(value) produces infinity
                double expSum = 0;

                double[] outputBeforeSoftmax = new double[NumNodes];
                double max = 0;

                for (int i = 0; i < NumNodes; i++)
                {
                    for (int j = 0; j < NumWeightsPerNode; j++)
                    {
                        outputBeforeSoftmax[i] += Weights[i * NumWeightsPerNode + j] * (j == 0 ? 1 : input[j - 1]);
                    }

                    //Keep track of maximum
                    if (outputBeforeSoftmax[i] > max) max = outputBeforeSoftmax[i];
                }

                for (int i = 0; i < NumNodes; i++)
                {
                    Outputs[i] = Math.Exp(outputBeforeSoftmax[i] - max);

                    expSum += Outputs[i];
                }

                for (int i = 0; i < NumNodes; i++)
                {
                    Outputs[i] /= (expSum);
                }

                return Outputs;
            }
            else
            {
                return base.Evaluate(input);
            }
        }

        public double Backpropagate(LearningMethod learningMethod, double[] actual, double[] expected, double[] nextOutputs, double learningRate, double momentum, double weightDecay, MiniBatchMode miniBatchMode)
        {
            double error = 0;

            if (miniBatchMode == MiniBatchMode.Off || miniBatchMode == MiniBatchMode.First)
            {
                ResetWeightGradients();
            }

            if (ActivationFuncType == ActivationFunctionType.Softmax)
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    /* Backpropagation:
                     * 
                     *   dE/dO for softmax with cross-entropy error is just t - y
                     */
                    OutputGradients[i] = (expected[i] - actual[i]);

                    //error += Math.Abs(expected[i] - actual[i]); //this isnt right, error function should be different for softmax
                    error += -expected[i] * Math.Log(actual[i]) + (1 - expected[i]) * Math.Log(1 - actual[i]);
                }

                error /= expected.Length;
            }
            else
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    /* Backprogation:
                     * 
                     *   dE/dO(i) = activation_derivative(O(i)) * error_func_derivative(O(i))
                     * 
                     * derivative of squared error is just t - y
                     */
                    OutputGradients[i] = (expected[i] - actual[i]) * ActivationFunctions.Get(ActivationFuncType).Derivative(actual[i]);

                    error += Math.Abs(expected[i] - actual[i]);
                }
            }

            UpdateWeightGradients(nextOutputs);
            UpdateWeights(learningMethod, miniBatchMode, learningRate, momentum, weightDecay);

            return error;
        }
    }
}
