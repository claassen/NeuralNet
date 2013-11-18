using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;
using System.IO;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetwork
    {
        public string Name;
        public InputLayer InputLayer;

        [XmlArrayItem(Type = typeof(HiddenLayer)),
         XmlArrayItem(Type = typeof(ConvolutionalLayer))]
        public List<HiddenLayer> HiddenLayers; //index 0 is closest to input

        public OutputLayer OutputLayer;
        
        public double LearningRate;
        
        private Random m_Rand;

        private const double MOMENTUM = 0.5;

        //So serialization works
        public NeuralNetwork() { }

        public NeuralNetwork(string networkName, InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer, double learningRate)
        {
            Name = networkName;
            InputLayer = inputLayer;
            HiddenLayers = hiddenLayers;
            OutputLayer = outputLayer;
            LearningRate = learningRate;

            m_Rand = new Random(DateTime.Now.Millisecond);

            Init();
            RandomizeWeights();
        }

        private void Init()
        {
            Layer prevLayer = InputLayer;
            Layer layer;

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                layer = HiddenLayers[i];
                layer.Init(prevLayer.NumNodes);
                if (layer is ConvolutionalLayer)
                {
                    ((ConvolutionalLayer)layer).InitFeatureMaps(prevLayer.NumFeatureMaps);
                }
                
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
        public double Train(double[] input, double[] expected)
        {
            double[] actual = GetResult(input);

            //Output nodes
            double error = OutputLayer.Backpropagate(actual, expected, HiddenLayers.Last().Outputs, LearningRate, MOMENTUM);

            Layer previous = OutputLayer;

            //Hidden nodes
            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                Layer layer = HiddenLayers[i];
                Layer next = (i == 0 ? (Layer)InputLayer : (Layer)HiddenLayers[i - 1]);

                layer.Backpropagate(previous, next, LearningRate, MOMENTUM);

                previous = layer;
            }

            return error;
        }

        public void SaveToDisk(string directory)
        {
            Serialization.SerializeObject<NeuralNetwork>(this, Path.Combine(directory, Name + ".xml"));
        }

        public void LoadFromDisk(string directory)
        {
            NeuralNetwork temp = Serialization.DeSerializeObject<NeuralNetwork>(Path.Combine(directory, Name + ".xml"));

            this.InputLayer = temp.InputLayer;
            this.HiddenLayers = temp.HiddenLayers;
            this.OutputLayer = temp.OutputLayer;
            this.LearningRate = temp.LearningRate;

            foreach (HiddenLayer layer in HiddenLayers)
            {
                if (layer is ConvolutionalLayer)
                {
                    ConvolutionalLayer cLayer = layer as ConvolutionalLayer;
                    foreach (FeatureMap fm in cLayer.FeatureMaps)
                    {
                        fm.Layer = cLayer;
                    }
                }
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
                    Console.Write(Math.Round(OutputLayer.Weights[i * OutputLayer.NumWeightsPerNode + j], 3, MidpointRounding.AwayFromZero) + " ");
                }
                Console.Write("\nGRADIENTS: ");
                for (int j = 0; j < OutputLayer.NumWeightsPerNode; j++)
                {
                    Console.Write(Math.Round(OutputLayer.WeightGradients[i * OutputLayer.NumWeightsPerNode + j], 3, MidpointRounding.AwayFromZero) + " ");
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
                        Console.Write(Math.Round(layer.Weights[i * layer.NumWeightsPerNode + j], 3, MidpointRounding.AwayFromZero) + " ");
                    }
                    Console.Write("\nGRADIENTS: ");
                    for (int j = 0; j < layer.NumWeightsPerNode; j++)
                    {
                        Console.Write(Math.Round(layer.WeightGradients[i * layer.NumWeightsPerNode + j], 3, MidpointRounding.AwayFromZero) + " ");
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
