using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;
using System.IO;
using NeuralNetwork.Layers;
using NeuralNetwork.Utils;
using NeuralNetwork.DataSetProviders;

namespace NeuralNetwork
{
    public enum LearningMethod
    {
        SGD = 0,
        RMSPROP = 1
    }

    public enum MiniBatchMode
    {
        Off = 0,
        First = 1,
        Accumulate = 2,
        Compute = 3
    }

    [Serializable]
    public class NeuralNetwork
    {
        public string Name;

        public InputLayer InputLayer;

        [XmlArrayItem(Type = typeof(FullyConnectedLayer)),
         XmlArrayItem(Type = typeof(ConvolutionalLayer))]
        public List<HiddenLayer> HiddenLayers; //index 0 is closest to input

        public OutputLayer OutputLayer;

        public LearningMethod LearningMethod;
        public double LearningRate;
        public double Momentum;
        public double WeightDecay;

        private Random m_Rand;

        //So serialization works
        public NeuralNetwork() { }

        public NeuralNetwork(IDataSetProvider dataSetProvider, string networkName, InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer/*, double learningRate, double momentum, double weightDecay*/)
        {
            Name = networkName;
            InputLayer = inputLayer;
            HiddenLayers = hiddenLayers;
            OutputLayer = outputLayer;
            
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

        public double Train(TrainingExample example, MiniBatchMode miniBatchMode = MiniBatchMode.Off)
        {
            double[] actual = GetResult(example.Input);

            //Output nodes
            double error = OutputLayer.Backpropagate(LearningMethod, actual, example.Expected, HiddenLayers.Last().Outputs, LearningRate, Momentum, WeightDecay, miniBatchMode);

            Layer previous = OutputLayer;

            //Hidden nodes
            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                Layer layer = HiddenLayers[i];
                Layer next = (i == 0 ? (Layer)InputLayer : (Layer)HiddenLayers[i - 1]);

                layer.Backpropagate(LearningMethod, previous, next, LearningRate, Momentum, WeightDecay, miniBatchMode);

                previous = layer;
            }

            return error;
        }

        public void SaveToDisk(string path)
        {
            Serialization.SerializeObject<NeuralNetwork>(this, path);
        }

        public void LoadFromDisk(string path)
        {
            NeuralNetwork temp = Serialization.DeSerializeObject<NeuralNetwork>(path);

            this.InputLayer = temp.InputLayer;
            this.HiddenLayers = temp.HiddenLayers;
            this.OutputLayer = temp.OutputLayer;
            this.LearningRate = temp.LearningRate;
            this.Momentum = temp.Momentum;
            this.WeightDecay = temp.WeightDecay;

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
    }
}
