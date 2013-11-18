using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace NeuralNetwork
{
    [Serializable]
    public class Layer
    {
        public ActivationFunctionType ActivationFuncType;
        public int NumNodes { get; set; }
        public int NumWeightsPerNode { get; set; }
        public double[] Weights { get; set; }
        public int NumFeatureMaps { get; set; }

        //[XmlIgnore, NonSerialized]
        public double[] Outputs;
        //[XmlIgnore, NonSerialized]
        public double[] OutputGradients;
        //[XmlIgnore, NonSerialized]
        public double[] WeightGradients;
        //[XmlIgnore, NonSerialized]
        public double[] PrevWeightGradients;
        
        public Layer() { }

        public Layer(ActivationFunctionType activationFunctionType, int numNodes)
        {
            ActivationFuncType = activationFunctionType;
            NumNodes = numNodes;
            Outputs = new double[NumNodes];
            OutputGradients = new double[NumNodes];
        }

        public virtual void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = numPreviousLayer + 1;
            Weights = new double[NumNodes * NumWeightsPerNode];
            WeightGradients = new double[NumNodes * NumWeightsPerNode];
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
                    if (Double.IsNaN(Weights[i * NumWeightsPerNode + j]))
                    {
                        Console.WriteLine("asd");
                    }

                    sum += Weights[i * NumWeightsPerNode + j] * (j == 0 ? 1 : input[j - 1]);
                }
                Outputs[i] = ActivationFunctions.Get(ActivationFuncType).Function(sum);

                if (Double.IsNaN(Outputs[i]))
                {
                    Console.WriteLine("asd");
                }

            }

            return Outputs;
        }

        public virtual void RandomizeWeights()
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < NumNodes; i++)
            {
                for (int j = 0; j < NumWeightsPerNode; j++)
                {
                    Weights[i * NumWeightsPerNode + j] = rand.NextDouble() * 2 - 1;
                }
            }
        }

        public virtual double Backpropagate(Layer previous, Layer next, double learningRate, double momentum)
        {
            throw new NotImplementedException();
        }
    }

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

        public double Backpropagate(double[] actual, double[] expected, double[] nextOutputs, double learningRate, double momentum)
        {
            double error = 0;

            if (ActivationFuncType == ActivationFunctionType.Softmax)
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    OutputGradients[i] = (expected[i] - actual[i]) * NumNodes;

                    if (Double.IsNaN(OutputGradients[i]))
                    {
                        Console.WriteLine("asd");
                    }

                    error += Math.Abs(expected[i] - actual[i]);

                    for (int j = 0; j < NumWeightsPerNode; j++)
                    {
                        WeightGradients[i * NumWeightsPerNode + j] = learningRate * OutputGradients[i] * (j == 0 ? 1 : nextOutputs[j - 1]);

                        //Weight decay
                        WeightGradients[i * NumWeightsPerNode + j] -= 0.1 * learningRate * Weights[i * NumWeightsPerNode + j];

                        Weights[i * NumWeightsPerNode + j] += WeightGradients[i * NumWeightsPerNode + j];
                        Weights[i * NumWeightsPerNode + j] += momentum * PrevWeightGradients[i * NumWeightsPerNode + j];
                        PrevWeightGradients[i * NumWeightsPerNode + j] = WeightGradients[i * NumWeightsPerNode + j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    OutputGradients[i] = (expected[i] - actual[i]) * ActivationFunctions.Get(ActivationFuncType).Derivative(actual[i]) * NumNodes;

                    error += Math.Abs(expected[i] - actual[i]);

                    for (int j = 0; j < NumWeightsPerNode; j++)
                    {
                        WeightGradients[i * NumWeightsPerNode + j] = learningRate * OutputGradients[i] * (j == 0 ? 1 : nextOutputs[j - 1]);

                        //Weight decay
                        WeightGradients[i * NumWeightsPerNode + j] -= 0.1 * learningRate * Weights[i * NumWeightsPerNode + j];

                        Weights[i * NumWeightsPerNode + j] += WeightGradients[i * NumWeightsPerNode + j];
                        Weights[i * NumWeightsPerNode + j] += momentum * PrevWeightGradients[i * NumWeightsPerNode + j];
                        PrevWeightGradients[i * NumWeightsPerNode + j] = WeightGradients[i * NumWeightsPerNode + j];
                    }
                }
            }

            return error;
        }
    }

    public class HiddenLayer : Layer
    {
        public HiddenLayer() { }

        public HiddenLayer(ActivationFunctionType activationFunctionType, int numNodes)
            : base(activationFunctionType, numNodes)
        {
        }

        public override double Backpropagate(Layer previous, Layer next, double learningRate, double momentum)
        {
            for (int j = 0; j < NumNodes; j++)
            {
                double outputGradientSum = 0;
                for (int k = 0; k < previous.OutputGradients.Length; k++)
                {
                    outputGradientSum += previous.OutputGradients[k] * Outputs[j];
                }

                OutputGradients[j] = ActivationFunctions.Get(ActivationFuncType).Derivative(Outputs[j]) * outputGradientSum;

                if (Double.IsNaN(OutputGradients[j]))
                {
                    Console.WriteLine("asd");
                }

                int weightIndexBase = j * NumWeightsPerNode;
                double lrTimesGradient = learningRate * OutputGradients[j];

                for (int k = 0; k < NumWeightsPerNode; k++)
                {
                    int weightIndex = weightIndexBase + k;
                    WeightGradients[weightIndex] = lrTimesGradient * (k == 0 ? 1 : next.Outputs[k - 1]);

                    //Weight decay
                    WeightGradients[weightIndex] -= 0.1 * lrTimesGradient * Weights[weightIndex];

                    Weights[weightIndex] += WeightGradients[weightIndex];
                    Weights[weightIndex] += momentum * PrevWeightGradients[weightIndex];
                    PrevWeightGradients[weightIndex] = WeightGradients[weightIndex];
                }
            }

            return 0;
        }
    }

    public class ConvolutionalLayer : HiddenLayer
    {
        public int KernelWidth;
        public int FeatureMapWidth;
        public int StepSize;
        public int NumPrevFeatureMaps;
        public bool[] ConnectionMap;
        public FeatureMap[] FeatureMaps;

        private int NumPreviousLayer;

        public ConvolutionalLayer() { }

        public ConvolutionalLayer(ActivationFunctionType activationFunctionType, int kernelWidth, int numFeatureMaps, int stepSize, bool[] connectionMap = null)
            : base(activationFunctionType, 0)
        {
            KernelWidth = kernelWidth;
            NumFeatureMaps = numFeatureMaps;
            StepSize = stepSize;
            ConnectionMap = connectionMap;
        }

        public override void Init(int numPreviousLayer)
        {
            NumPreviousLayer = numPreviousLayer;
        }

        public void InitFeatureMaps(int numPrevFeatureMaps)
        {
            NumPrevFeatureMaps = numPrevFeatureMaps;

            if (ConnectionMap == null && NumPrevFeatureMaps > 1)
            {
                Random rand = new Random(DateTime.Now.Millisecond);

                ConnectionMap = new bool[NumPrevFeatureMaps * NumFeatureMaps];
                for (int i = 0; i < NumPrevFeatureMaps * NumFeatureMaps; i++)
                {
                    ConnectionMap[i] = rand.NextDouble() > 0.8 ? true : false;
                }
            }

            //This is tricky, figure out what our feature map size should be based on the previous feature
            //map size, kernel size and step size
            FeatureMapWidth = ((int)Math.Sqrt(NumPreviousLayer / numPrevFeatureMaps) - KernelWidth) / StepSize + 1;

            NumNodes = NumFeatureMaps * FeatureMapWidth * FeatureMapWidth;
            Outputs = new double[NumNodes];
            OutputGradients = new double[NumNodes];

            FeatureMaps = new FeatureMap[NumFeatureMaps];

            for (int i = 0; i < NumFeatureMaps; i++)
            {
                FeatureMaps[i] = new FeatureMap(this);   
            }
        }

        public override double[] Evaluate(double[] input)
        {
            int prevFMSize = input.Length / NumPrevFeatureMaps;

            //Split input into feature maps to avoid array index arithmetic hell
            double[][] inputAsFeatureMaps = new double[NumPrevFeatureMaps][];
            for (int i = 0; i < NumPrevFeatureMaps; i++)
            {
                inputAsFeatureMaps[i] = new double[prevFMSize];
                
                for(int j = 0; j < prevFMSize; j++)
                {
                    inputAsFeatureMaps[i][j] = input[i * prevFMSize + j];
                }
            }

            //Convolute the feature maps
            for (int i = 0; i < NumFeatureMaps; i++)
            {
                for (int j = 0; j < NumPrevFeatureMaps; j++)
                {
                    if (FMIsConnectedToPrevFM(i, j))
                    {
                        FeatureMaps[i].Convolute(inputAsFeatureMaps[j], j);
                    }
                }
            }

            //Extract the output from the feature maps (as we provide a standard interface of returning output as a double[]) and add bias
            for (int i = 0; i < NumFeatureMaps; i++)
            {
                double[] fmOutput = FeatureMaps[i].Output;

                for (int j = 0; j < FeatureMapWidth * FeatureMapWidth; j++)
                {
                    Outputs[j + i * FeatureMapWidth * FeatureMapWidth] = ActivationFunctions.Get(ActivationFuncType).Function(fmOutput[j] + FeatureMaps[i].Bias);
                }
            }

            return Outputs;
        }

        public override double Backpropagate(Layer previous, Layer next, double learningRate, double momentum)
        {
            if (previous is ConvolutionalLayer)
            {
                ConvolutionalLayer prevCLayer = (ConvolutionalLayer)previous;

                for (int i = 0; i < NumFeatureMaps * FeatureMapWidth * FeatureMapWidth; i++)
                {
                    OutputGradients[i] = 0;
                }

                //Calculate output gradient sums
                for (int fm = 0; fm < NumFeatureMaps; fm++)
                {
                    //pixel 0 for the current feature map in this layers output
                    int nodeIndexFMBase = fm * FeatureMapWidth * FeatureMapWidth;

                    //weight 0 of the kernel weights for the current FM in this layer in the current FM in previous layer
                    int prevKernelWeightIndexBase = fm * prevCLayer.KernelWidth * prevCLayer.KernelWidth; 

                    for (int prevFm = 0; prevFm < prevCLayer.NumFeatureMaps; prevFm++)
                    {
                        if (!prevCLayer.FMIsConnectedToPrevFM(prevFm, fm))
                        {
                            continue;
                        }

                        //pixel 0 for the current previous FM in the previous output
                        int prevNodeIndexBase = prevFm * prevCLayer.FeatureMapWidth * prevCLayer.FeatureMapWidth;

                        for (int fy = 0; fy < prevCLayer.FeatureMapWidth; fy++) //each iteration moves the kernel 2 pixels to the right
                        {
                            for (int fx = 0; fx < prevCLayer.FeatureMapWidth; fx++) //each iteration moves the kernel 2 pixels down
                            {
                                //current pixel of the previous FM in the previous output, corresponds to the output of applying the kernel in its current position
                                int prevNodeIndex = prevNodeIndexBase + fx + fy * prevCLayer.FeatureMapWidth;

                                //top left corner of the current kernel position over this layers output
                                //the 2*s are because the kernel moves 2 pixels at a time across the input image
                                int nodeIndexBase = nodeIndexFMBase + StepSize * fx + StepSize * (fy * FeatureMapWidth);
                                //int nodeIndexBase = nodeIndexFMBase + fx * StepSize +  
                                for (int ky = 0; ky < prevCLayer.KernelWidth; ky++)
                                {
                                    for (int kx = 0; kx < prevCLayer.KernelWidth; kx++)
                                    {
                                        //position in this layers output of that pixel that connects to the current node in the kernel at its current position
                                        int nodeIndex = nodeIndexBase + kx + ky * prevCLayer.KernelWidth;

                                        double dEY = prevCLayer.OutputGradients[prevNodeIndex];

                                        OutputGradients[nodeIndex] += dEY * prevCLayer.FeatureMaps[prevFm].KernelWeights[prevKernelWeightIndexBase + kx + ky * prevCLayer.KernelWidth];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    double outputGradientSum = 0;
                    for (int k = 0; k < previous.OutputGradients.Length; k++)
                    {
                        outputGradientSum += previous.OutputGradients[k] * Outputs[i];
                    }

                    OutputGradients[i] = outputGradientSum ;
                }
            }

            //Calculate feature map biases by adding up gradients of each pixel in that feature map
            for (int i = 0; i < NumFeatureMaps; i++)
            {
                int nodeIndexBase = i * FeatureMapWidth * FeatureMapWidth;
                double biasGradient = 0;

                for (int j = 0; j < FeatureMapWidth; j++)
                {
                    biasGradient += OutputGradients[nodeIndexBase + j];
                }

                FeatureMaps[i].BiasGradient = biasGradient;
                FeatureMaps[i].Bias += learningRate * biasGradient;

                //Add momentum
                FeatureMaps[i].Bias += momentum * FeatureMaps[i].PrevBiasGradient;
                FeatureMaps[i].PrevBiasGradient = biasGradient;
            }

            //Compute final output gradients by multiplying them by the derivative of the activation function
            for (int i = 0; i < NumNodes; i++)
            {
                OutputGradients[i] = ActivationFunctions.Get(ActivationFuncType).Derivative(Outputs[i]) * OutputGradients[i];
            }

            //Split output into feature maps to /*avoid*/ marginally minimize array index arithmetic hell
            double[][] outputGradientsAsFeatureMapOutputGradients = new double[NumFeatureMaps][];
            for (int i = 0; i < NumFeatureMaps; i++)
            {
                outputGradientsAsFeatureMapOutputGradients[i] = new double[FeatureMapWidth * FeatureMapWidth];
                for (int j = 0; j < FeatureMapWidth * FeatureMapWidth; j++)
                {
                    outputGradientsAsFeatureMapOutputGradients[i][j] = OutputGradients[i * FeatureMapWidth * FeatureMapWidth + j];
                }
            }

            int nextFeatureMapWidth = (int)Math.Sqrt(next.Outputs.Length / next.NumFeatureMaps);
            for (int i = 0; i < NumFeatureMaps; i++)
            {
                FeatureMaps[i].Backpropagate(next.Outputs, nextFeatureMapWidth, outputGradientsAsFeatureMapOutputGradients[i], learningRate, momentum, i);
            }

            return 0;
        }

        public override void RandomizeWeights()
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < FeatureMaps.Length; i++)
            {
                FeatureMaps[i].RandomizeWeights(rand);
            }
        }

        public bool FMIsConnectedToPrevFM(int thisFMIndex, int prevFMIndex)
        {
            return null == ConnectionMap || ConnectionMap[prevFMIndex * NumFeatureMaps + thisFMIndex];
        }
    }

    public class InputLayer : Layer
    {
        public InputLayer() { }

        public InputLayer(int numNodes)
            : base(ActivationFunctionType.Sigmoid, numNodes)
        {
            Init(0);
        }

        public override void Init(int numPreviousLayer)
        {
            NumWeightsPerNode = 1;
            NumFeatureMaps = 1;
        }

        public override double[] Evaluate(double[] input)
        {
            Outputs = input;
            return Outputs; ;
        }
    }
}
