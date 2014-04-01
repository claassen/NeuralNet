using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Functions;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    public class ConvolutionalLayer : HiddenLayer
    {
        public int KernelWidth;
        public int FeatureMapWidth;
        public int StepSize;
        public int NumPrevFeatureMaps;
        public int NumWeightsPerFeatureMap;
        public bool UseRandomConnectionsToPrev;
        public double RandomConnectionDensity;
        public bool[] ConnectionMap;
        public FeatureMap[] FeatureMaps;

        private int NumPreviousLayer;

        public ConvolutionalLayer() { }

        public ConvolutionalLayer(ActivationFunctionType activationFunctionType, int kernelWidth, int numFeatureMaps, int stepSize, bool useRandomConnectionsToPrev, double randomConnectionDensity)
            : base(activationFunctionType, 0)
        {
            KernelWidth = kernelWidth;
            NumFeatureMaps = numFeatureMaps;
            StepSize = stepSize;
            UseRandomConnectionsToPrev = useRandomConnectionsToPrev;
            RandomConnectionDensity = randomConnectionDensity;
        }

        public override void Init(int numPreviousLayer)
        {
            NumPreviousLayer = numPreviousLayer;
        }

        public void InitFeatureMaps(int numPrevFeatureMaps)
        {
            NumPrevFeatureMaps = numPrevFeatureMaps;

            //Set up random connections to previous layer feature maps
            if (UseRandomConnectionsToPrev && NumPrevFeatureMaps > 1)
            {
                Random rand = new Random(DateTime.Now.Millisecond);

                ConnectionMap = new bool[NumPrevFeatureMaps * NumFeatureMaps];

                for (int i = 0; i < NumPrevFeatureMaps * NumFeatureMaps; i++)
                {
                    ConnectionMap[i] = rand.NextDouble() > (1 - RandomConnectionDensity) ? true : false;
                }
            }

            //This is tricky, figure out what our feature map size should be based on the previous feature
            //map size, kernel size and step size
            FeatureMapWidth = ((int)Math.Sqrt(NumPreviousLayer / numPrevFeatureMaps) - KernelWidth) / StepSize + 1;

            NumNodes = NumFeatureMaps * FeatureMapWidth * FeatureMapWidth;
            NumWeightsPerFeatureMap = NumPrevFeatureMaps * KernelWidth * KernelWidth + 1;
            NumWeightsPerNode = (NumWeightsPerFeatureMap * NumFeatureMaps) / NumNodes;
            Outputs = new double[NumNodes];
            OutputGradients = new double[NumNodes];
            Weights = new double[NumFeatureMaps * NumWeightsPerFeatureMap];
            WeightGradients = new double[NumFeatureMaps * NumWeightsPerFeatureMap];
            WeightGradientMeanSquares = new double[NumFeatureMaps * NumWeightsPerFeatureMap];
            PrevWeightGradients = new double[NumFeatureMaps * NumWeightsPerFeatureMap];

            FeatureMaps = new FeatureMap[NumFeatureMaps];

            for (int i = 0; i < NumFeatureMaps; i++)
            {
                FeatureMaps[i] = new FeatureMap(this, i);
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

                for (int j = 0; j < prevFMSize; j++)
                {
                    inputAsFeatureMaps[i][j] = input[i * prevFMSize + j];
                }
            }

            Parallel.For(0, NumFeatureMaps, index =>
            {
                for (int j = 0; j < NumPrevFeatureMaps; j++)
                {
                    if (FMIsConnectedToPrevFM(index, j))
                    {
                        FeatureMaps[index].Convolute(inputAsFeatureMaps[j], j);
                    }
                }
            });

            Parallel.For(0, NumFeatureMaps, index =>
            {
                double[] fmOutput = FeatureMaps[index].Output;

                for (int j = 0; j < FeatureMapWidth * FeatureMapWidth; j++)
                {
                    Outputs[j + index * FeatureMapWidth * FeatureMapWidth] = fmOutput[j];
                }
            });

            return Outputs;
        }

        public override double Backpropagate(LearningMethod learningMethod, Layer previous, Layer next, double learningRate, double momentum, double weightDecay, MiniBatchMode miniBatchMode)
        {
            if (previous is ConvolutionalLayer)
            {
                ConvolutionalLayer prevCLayer = (ConvolutionalLayer)previous;

                for (int i = 0; i < NumFeatureMaps * FeatureMapWidth * FeatureMapWidth; i++)
                {
                    OutputGradients[i] = 0;
                }

                //Calculate output gradient sums
                Parallel.For(0, NumFeatureMaps, fm =>
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
                });
            }
            else
            {
                for (int i = 0; i < NumNodes; i++)
                {
                    double outputGradientSum = 0;
                    for (int k = 0; k < previous.OutputGradients.Length; k++)
                    {
                        outputGradientSum += previous.OutputGradients[k] * previous.Weights[k * previous.NumWeightsPerNode + i];
                    }

                    OutputGradients[i] = outputGradientSum;
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

                int biasWeightIndex = (i + 1) * NumWeightsPerFeatureMap - 1;

                WeightGradients[biasWeightIndex] = biasGradient;
                
                //FeatureMaps[i].BiasGradient = biasGradient;
                //FeatureMaps[i].Bias += learningRate * biasGradient;

                //Add momentum
                //FeatureMaps[i].Bias += momentum * FeatureMaps[i].PrevBiasGradient;
                //FeatureMaps[i].PrevBiasGradient = biasGradient;
            }

            //Compute final output gradients by multiplying them by the derivative of the activation function
            for (int i = 0; i < NumNodes; i++)
            {
                OutputGradients[i] *= ActivationFunctions.Get(ActivationFuncType).Derivative(Outputs[i]);
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

            Parallel.For(0, NumFeatureMaps, i =>
            {
                FeatureMaps[i].Backpropagate(next.Outputs, nextFeatureMapWidth, outputGradientsAsFeatureMapOutputGradients[i], learningRate, momentum, weightDecay, miniBatchMode);
            });

            UpdateWeights(learningMethod, miniBatchMode, learningRate, momentum, weightDecay);

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
}
