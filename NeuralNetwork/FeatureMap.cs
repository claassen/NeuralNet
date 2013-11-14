using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace NeuralNetwork
{
    [Serializable]
    public class FeatureMap
    {
        public int Width;
        public int KernelWidth;
        public int NumKernels;
        public double[] KernelWeights;
        public double Bias;

        [XmlIgnore, NonSerialized]
        public double BiasGradient;
        [XmlIgnore, NonSerialized]
        public double PrevBiasGradient;
        [XmlIgnore, NonSerialized]
        public double[] KernelWeightGradents;
        [XmlIgnore, NonSerialized]
        public double[] PrevKernelWeightGradients;
        [XmlIgnore, NonSerialized]
        public double[] Output;

        public FeatureMap() { }

        public FeatureMap(int width, int kernelWidth, int numKernels)
        {
            Width = width;
            KernelWidth = kernelWidth;
            NumKernels = numKernels;
            KernelWeights = new double[NumKernels * KernelWidth * KernelWidth];
            KernelWeightGradents = new double[NumKernels * KernelWidth * KernelWidth];
            PrevKernelWeightGradients = new double[NumKernels * KernelWidth * KernelWidth];
            Output = new double[Width * Width];
        }

        public void Convolute(double[] input, int index)
        {
            int inputFMWidth = (int)Math.Sqrt(input.Length);
            int kernelWeightIndexBase = index * KernelWidth * KernelWidth;

            for (int y = 0; y < Width; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    for (int ky = 0; ky < KernelWidth; ky++)
                    {
                        for (int kx = 0; kx < KernelWidth; kx++)
                        {
                            Output[x + y * Width] += input[kx + x * 2 + ky * inputFMWidth + y * (inputFMWidth * 2)] * KernelWeights[kernelWeightIndexBase + kx + ky * KernelWidth];
                        }
                    }
                }
            }
        }

        public void Backpropagate(double[] nextOutputs, int nextFeatureMapWidth, double[] fmOutputGradients, double learningRate, double momentum)
        {
            int nextOutputWidth = (int)Math.Sqrt(nextOutputs.Length);

            for (int k = 0; k < NumKernels; k++)
            {
                //pixel 0 of the next layer feature map associated with the current kernel
                int nextNodeIndexFMBase = k * nextFeatureMapWidth * nextFeatureMapWidth;

                for (int y = 0; y < Width; y++)
                {
                    for(int x = 0; x < Width; x++)
                    {
                        //previously calculated error gradient of the current feature map pixel
                        double dErrY = fmOutputGradients[x + y * Width];

                        //top left corner at the current kernel position in the next output
                        int nextNodeIndexBase = nextNodeIndexFMBase + 2 * x + 2 * nextOutputWidth * y;

                        for (int ky = 0; ky < KernelWidth; ky++)
                        {
                            for(int kx = 0; kx < KernelWidth; kx++)
                            {
                                KernelWeightGradents[k * KernelWidth * KernelWidth + kx + ky * KernelWidth] += dErrY * nextOutputs[nextNodeIndexBase + kx + ky * nextFeatureMapWidth];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < NumKernels; i++)
            {
                for (int j = 0; j < KernelWidth * KernelWidth; j++)
                {
                    KernelWeights[i * KernelWidth * KernelWidth + j] += learningRate * KernelWeightGradents[i * KernelWidth * KernelWidth + j];

                    //Add momentum
                    KernelWeights[i * KernelWidth * KernelWidth + j] += momentum * PrevKernelWeightGradients[i * KernelWidth * KernelWidth + j];
                    PrevKernelWeightGradients[i * KernelWidth * KernelWidth + j] = KernelWeightGradents[i * KernelWidth * KernelWidth + j];
                }
            }
        }

        public void RandomizeWeights(Random rand)
        {
            Bias = rand.NextDouble() * 2 - 1;

            for (int i = 0; i < NumKernels; i++)
            {
                for (int j = 0; j < NumKernels * KernelWidth * KernelWidth; j++)
                {
                    KernelWeights[i * KernelWidth * KernelWidth + j] = rand.NextDouble() * 2 - 1;
                }
            }
        }
    }
}
