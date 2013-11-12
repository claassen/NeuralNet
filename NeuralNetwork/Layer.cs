using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Layer
    {
        public ActivationFunction ActivationFunction { get; set; }
        public int NumNodes { get; set; }
        public int NumWeightsPerNode { get; set; }
        public double[] Outputs { get; set; }
        public double[] OutputGradients { get; set; }
        public double[,] Weights { get; set; }
        public double[,] WeightGradients { get; set; }
        public double[,] PrevWeightGradients { get; set; }

        public Layer(ActivationFunction activationFunction, int numNodes)
        {
            ActivationFunction = activationFunction;
            NumNodes = numNodes;
            Outputs = new double[NumNodes];
            OutputGradients = new double[NumNodes];
        }

        public virtual void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = numPreviousLayer + 1;
            Weights = new double[NumNodes, NumWeightsPerNode];
            WeightGradients = new double[NumNodes, NumWeightsPerNode];
            PrevWeightGradients = new double[NumNodes, NumWeightsPerNode];
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
                    sum += Weights[i, j] * (j == 0 ? 1 : input[j - 1]);
                }
                Outputs[i] = ActivationFunction.Function(sum);
            }

            return Outputs;
        }

        public void RandomizeWeights()
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < NumNodes; i++)
            {
                for (int j = 0; j < NumWeightsPerNode; j++)
                {
                    Weights[i, j] = rand.NextDouble() * 2 - 1;
                }
            }
        }
    }

    public class OutputLayer : Layer
    {
        public OutputLayer(ActivationFunction activationFunction, int numNodes) 
            : base(activationFunction, numNodes)
        {
        }
    }

    public class HiddenLayer : Layer
    {
        public int Type { get; set; } //unused

        public HiddenLayer(ActivationFunction activationFunction, int numNodes)
            : base(activationFunction, numNodes)
        {
        }
    }

    public class InputLayer : Layer
    {
        public InputLayer(int numNodes)
            : base(null, numNodes)
        {
            Init(0);
        }

        public override void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = 1;
        }

        public override double[] Evaluate(double[] input)
        {
            Outputs = input;
            return Outputs; ;
        }
    }
}
