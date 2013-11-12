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
            double[] result = InputLayer.Evaluate(input);

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                result = HiddenLayers[i].Evaluate(result);
            }

            return OutputLayer.Evaluate(result);
        }

        /*
         * Trains the neural network using the given training example. ("Online" training)
         */
        public void Train(double[] input, double[] expected)
        {
            double[] actual = GetResult(input);

            //Output nodes
            OutputLayer.Backpropagate(actual, expected, HiddenLayers.Last().Outputs, m_LearningRate, MOMENTUM);
            
            Layer previous = OutputLayer;

            //Hidden nodes
            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                HiddenLayer layer = HiddenLayers[i];
                Layer next = (i == 0 ? (Layer)InputLayer : (Layer)HiddenLayers[i - 1]);

                layer.Backpropagate(previous.OutputGradients, next.Outputs, m_LearningRate, MOMENTUM);

                previous = layer;
            }
        }

        public void DisplayNetwork()
        {
            Console.WriteLine("\n::OUTPUT LAYER::\n");
            for (int i = 0; i < OutputLayer.NumNodes; i++)
            {
                Console.WriteLine("OUTPUT NODE " + (i + 1));
                Console.WriteLine("OUTPUT: " + Math.Round(OutputLayer.Outputs[i], 3, MidpointRounding.AwayFromZero));
                Console.Write("WEIGHTS: ");
                for (int j = 0; j < OutputLayer.NumWeightsPerNode; j++)
                {
                    Console.Write(Math.Round(OutputLayer.Weights[i, j], 3, MidpointRounding.AwayFromZero) + " ");
                }
                Console.Write("\nGRADIENTS: ");
                for (int j = 0; j < OutputLayer.NumWeightsPerNode; j++)
                {
                    Console.Write(Math.Round(OutputLayer.WeightGradients[i, j], 3, MidpointRounding.AwayFromZero) + " ");
                }
                Console.WriteLine("\n");
            }

            for (int k = HiddenLayers.Count - 1; k >= 0; k--)
            {
                HiddenLayer layer = HiddenLayers[k];
                Console.WriteLine("\n::HIDDEN LAYER " + (k + 1) + "::\n");
                for (int i = 0; i < layer.NumNodes; i++)
                {
                    Console.WriteLine("HIDDEN NODE " + (i + 1));
                    Console.WriteLine("OUTPUT: " + Math.Round(layer.Outputs[i], 3, MidpointRounding.AwayFromZero));
                    Console.Write("WEIGHTS: ");
                    for (int j = 0; j < layer.NumWeightsPerNode; j++)
                    {
                        Console.Write(Math.Round(layer.Weights[i, j], 3, MidpointRounding.AwayFromZero) + " ");
                    }
                    Console.Write("\nGRADIENTS: ");
                    for (int j = 0; j < layer.NumWeightsPerNode; j++)
                    {
                        Console.Write(Math.Round(layer.WeightGradients[i, j], 3, MidpointRounding.AwayFromZero) + " ");
                    }
                    Console.WriteLine("\n");
                }
            }

            Console.WriteLine("\n::INPUT LAYER::\n");
            Console.WriteLine("OUTPUTS (INPUTS):");
            for (int i = 0; i < InputLayer.NumNodes; i++)
            {
                Console.Write(Math.Round(InputLayer.Outputs[i], 3, MidpointRounding.AwayFromZero) + " ");
            }
            Console.WriteLine("\n");
        }
    }
}
