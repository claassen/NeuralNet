using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;
using System.Threading.Tasks;
using NeuralNetwork.Functions;

namespace NeuralNetwork.Layers
{
    [Serializable]
    public class Layer
    {
        public ActivationFunctionType ActivationFuncType;
        public int NumNodes { get; set; }
        public int NumWeightsPerNode { get; set; }
        public double[] Weights { get; set; }
        public int NumFeatureMaps { get; set; }

        public double[] PreActivationOutputs;
        public double[] Outputs;
        public double[] OutputGradients;
        public double[] WeightGradients;
        public double[] PrevWeightGradients;
        public double[] WeightGradientMeanSquares;
        
        public Layer() { }

        public Layer(ActivationFunctionType activationFunctionType, int numNodes)
        {
            ActivationFuncType = activationFunctionType;
            NumNodes = numNodes;
            Outputs = new double[NumNodes];
            PreActivationOutputs = new double[NumNodes];
            OutputGradients = new double[NumNodes];
        }

        protected void ResetWeightGradients()
        {
            for (int i = 0; i < NumNodes * NumWeightsPerNode; i++)
            {
                WeightGradients[i] = 0;
            }
        }

        public virtual void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = numPreviousLayer + 1;
            Weights = new double[NumNodes * NumWeightsPerNode];
            WeightGradients = new double[NumNodes * NumWeightsPerNode];
            WeightGradientMeanSquares = new double[NumNodes * NumWeightsPerNode];
            for (int i = 0; i < WeightGradientMeanSquares.Length; i++)
            {
                WeightGradientMeanSquares[i] = 1;
            }
            PrevWeightGradients = new double[NumNodes * NumWeightsPerNode];
            NumFeatureMaps = 1;
        }

        public virtual double[] Evaluate(double[] input)
        {
            if(input.Length != NumWeightsPerNode - 1)
                throw new ArgumentException("Input length does not match layer input size");

            for (int i = 0; i < NumNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < NumWeightsPerNode; j++)
                {
                    sum += Weights[i * NumWeightsPerNode + j] * (j == 0 ? 1 : input[j - 1]);
                }
                //PreActivationOutputs[i] = sum;
                Outputs[i] = ActivationFunctions.Get(ActivationFuncType).Function(sum);
            }

            return Outputs;
        }

        public virtual void RandomizeWeights()
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            //Fan-in is number of incomming connections to a node
            int fanIn = NumWeightsPerNode;

            for (int i = 0; i < NumNodes; i++)
            {
                for (int j = 0; j < NumWeightsPerNode; j++)
                {
                    Weights[i * NumWeightsPerNode + j] = (rand.NextDouble() * 2 - 1) * (1.0 / Math.Sqrt(fanIn));
                }
            }
        }

        public virtual double Backpropagate(LearningMethod learningMethod, Layer previous, Layer next, double learningRate, double momentum, double weightDecay, MiniBatchMode miniBatchMode)
        {
            throw new NotImplementedException();
        }

        protected void UpdateWeightGradients(double[] nextOutputs)
        {
            for (int nodeIndex = 0; nodeIndex < NumNodes; nodeIndex++)
            {
                for (int i = 0; i < NumWeightsPerNode; i++)
                {
                    int weightIndex = nodeIndex * NumWeightsPerNode + i;

                    /* Backpropagation:
                     * 
                     *   dE/dW(i->j) = O(i) * dE/dO(j)
                     *   
                     * where:
                     * i           = node index in previous layer
                     * j           = node index in this layer
                     * dE/dW(i->j) = the weight gradient for weight connecting output node i in the previous layer to node j in this layer
                     * O(i)        = output of node i in previous (lower) layer
                     * dE/dO(j)    = precalculated error gradient on the output of node j in this layer
                     */
                    WeightGradients[weightIndex] += OutputGradients[nodeIndex] * (i == 0 ? 1 : nextOutputs[i - 1]);
                }
            }
        }

        protected void UpdateWeights(LearningMethod learningMethod, MiniBatchMode miniBatchMode, double learningRate, double momentum, double weightDecay)
        {
            for (int n = 0; n < NumNodes; n++)
            {
                for (int i = 0; i < NumWeightsPerNode; i++)
                {
                    int weightIndex = n * NumWeightsPerNode + i;

                    if (miniBatchMode == MiniBatchMode.Off || miniBatchMode == MiniBatchMode.Compute)
                    {
                        if (learningMethod == LearningMethod.RMSPROP)
                        {
                            WeightGradientMeanSquares[weightIndex] = 0.9 * WeightGradientMeanSquares[weightIndex] + 
                                                                     0.1 * Math.Pow(WeightGradients[weightIndex], 2);

                            WeightGradients[weightIndex] /= Math.Sqrt(WeightGradientMeanSquares[weightIndex] + 1e-8);
                        }
                        else if (learningMethod == LearningMethod.SGD)
                        {
                            //Weight decay
                            //WeightGradients[weightIndex] -= weightDecay * learningRate * Weights[weightIndex];
                            //Weights[weightIndex] *= (1 - weightDecay);

                            //Momentum
                            Weights[weightIndex] += learningRate * momentum * PrevWeightGradients[weightIndex];
                            //WeightGradients[weightIndex] += momentum * PrevWeightGradients[weightIndex]; 
                        }
                        else
                        {
                            throw new InvalidOperationException("Invalid learning method specified.");
                        }

                        //Update the weight
                        Weights[weightIndex] += learningRate * WeightGradients[weightIndex];
                    }

                    PrevWeightGradients[weightIndex] = WeightGradients[weightIndex];
                }
            }
        }
    }
}
