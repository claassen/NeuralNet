using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public OutputLayer OutputLayer { get; set; }
        public List<HiddenLayer> HiddenLayers { get; set; }
        public InputLayer InputLayer { get; set; }

        private double m_LearningRate;
        private Random m_Rand;

        private const double MOMENTUM = 0.4;
        
        public NeuralNetwork(InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer, double learningRate)
        {
            InputLayer = inputLayer;
            HiddenLayers = hiddenLayers;
            OutputLayer = outputLayer;
            m_LearningRate = learningRate;

            m_Rand = new Random(DateTime.Now.Millisecond);

            Init();
            RandomizeWeights();
        }

        private void Init()
        {
            //Go from bottom up
            HiddenLayer layer = HiddenLayers[0];

            layer.Init(InputLayer.NumNodes);

            HiddenLayer prevLayer = layer;

            for (int i = 1; i < HiddenLayers.Count; i++)
            {
                layer = HiddenLayers[i];
                layer.Init(prevLayer.NumNodes);
                prevLayer = layer;
            }

            OutputLayer.Init(prevLayer.NumNodes);
        }

        public void RandomizeWeights()
        {
            //Output layer
            OutputLayer.RandomizeWeights();

            //Hidden layers
            foreach (HiddenLayer layer in HiddenLayers)
            {
                layer.RandomizeWeights();
            }
        }

        public double[] GetResult(double[] input)
        {
            double[] result = HiddenLayers[0].Evaluate(InputLayer.Evaluate(input));

            for (int i = 1; i < HiddenLayers.Count; i++)
            {
                result = HiddenLayers[i].Evaluate(result);
            }

            result = OutputLayer.Evaluate(result);

            return result;
        }

        /*
         * Trains the neural network using the given training example. ("Online" training)
         */
        public void Train(double[] input, double[] expected)
        {
            double[] actual = GetResult(input);

            //Output nodes
            for (int i = 0; i < OutputLayer.NumNodes; i++)
            {
                OutputLayer.OutputGradients[i] = (expected[i] - actual[i]) * OutputLayer.ActivationFunction.Derivative(actual[i]);

                for (int j = 0; j < OutputLayer.NumWeightsPerNode; j++)
                {
                    OutputLayer.WeightGradients[i, j] = m_LearningRate * OutputLayer.OutputGradients[i] * (j == 0 ? 1 : HiddenLayers.Last().Outputs[j - 1]);
                    OutputLayer.Weights[i, j] += OutputLayer.WeightGradients[i, j];
                    OutputLayer.Weights[i, j] += MOMENTUM * OutputLayer.PrevWeightGradients[i, j];
                    OutputLayer.PrevWeightGradients[i, j] = OutputLayer.WeightGradients[i, j];
                }
            }

            Layer previous = OutputLayer;

            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                Layer layer = HiddenLayers[i];
                Layer next = (i == 0 ? (Layer)InputLayer : (Layer)HiddenLayers[i - 1]);

                for (int j = 0; j < layer.NumNodes; j++)
                {
                    double outputGradientSum = 0;
                    for (int k = 0; k < previous.NumNodes; k++)
                    {
                        outputGradientSum += previous.OutputGradients[k] * layer.Outputs[j];
                    }

                    layer.OutputGradients[j] = layer.ActivationFunction.Derivative(layer.Outputs[j]) * outputGradientSum;

                    for (int k = 0; k < layer.NumWeightsPerNode; k++)
                    {
                        layer.WeightGradients[j, k] = m_LearningRate * layer.OutputGradients[j] * (k == 0 ? 1 : next.Outputs[k - 1]);
                        layer.Weights[j, k] += layer.WeightGradients[j, k];
                        layer.Weights[j, k] += MOMENTUM * layer.PrevWeightGradients[j, k];
                        layer.PrevWeightGradients[j, k] = layer.WeightGradients[j, k];
                    }
                }

                previous = layer;
            }
        }

        //public void PrettyDisplay()
        //{
        //    Console.WriteLine("\n::OUTPUT LAYER::\n");
        //    for (int i = 0; i < m_NumOutput; i++)
        //    {
        //        Console.WriteLine("OUTPUT NODE " + i + 1);
        //        Console.WriteLine("OUTPUT: " + m_OutputOutputs[i]);
        //        Console.Write("WEIGHTS: ");
        //        for (int j = 0; j < m_NumOutputWeights; j++)
        //        {
        //            Console.Write(m_OutputWeights[i,j] + " ");
        //        }
        //        Console.Write("\nGRADIENTS: ");
        //        for (int j = 0; j < m_NumOutputWeights; j++)
        //        {
        //            Console.Write(m_OutputWeightGradients[i,j] + " ");
        //        }
        //        Console.WriteLine("\n");
        //    }

        //    Console.WriteLine("\n::HIDDEN LAYER::\n");
        //    for (int i = 0; i < m_NumHidden; i++)
        //    {
        //        Console.WriteLine("HIDDEN NODE " + i + 1);
        //        Console.WriteLine("OUTPUT: " + m_HiddenOutputs[i]);
        //        Console.Write("WEIGHTS: ");
        //        for (int j = 0; j < m_NumHiddenWeights; j++)
        //        {
        //            Console.Write(m_HiddenWeights[i,j] + " ");
        //        }
        //        Console.Write("\nGRADIENTS: ");
        //        for (int j = 0; j < m_NumHiddenWeights; j++)
        //        {
        //            Console.Write(m_HiddenWeightGradients[i, j] + " ");
        //        }
        //        Console.WriteLine("\n");
        //    }
        //}
    }
}
