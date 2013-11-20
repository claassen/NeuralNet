using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    [Serializable]
    public class FeatureMap
    {
        [XmlIgnore, NonSerialized]
        public ConvolutionalLayer Layer;

        public double[] KernelWeights;
        public double Bias;
        public double BiasGradient;
        public double PrevBiasGradient;
        public double[] KernelWeightGradents;
        public double[] PrevKernelWeightGradients;
        public double[] Output;
       
        public FeatureMap() { }

        public FeatureMap(ConvolutionalLayer layer)
        {
            Layer = layer;
            KernelWeights = new double[Layer.NumPrevFeatureMaps * Layer.KernelWidth * Layer.KernelWidth];
            KernelWeightGradents = new double[Layer.NumPrevFeatureMaps * Layer.KernelWidth * Layer.KernelWidth];
            PrevKernelWeightGradients = new double[Layer.NumPrevFeatureMaps * Layer.KernelWidth * Layer.KernelWidth];
            Output = new double[Layer.FeatureMapWidth * Layer.FeatureMapWidth];
        }

        private void ResetWeightGradients()
        {
            for (int i = 0; i < KernelWeightGradents.Length; i++)
            {
                KernelWeightGradents[i] = 0;
            }
        }

        public void Convolute(double[] input, int index)
        {
            int inputFMWidth = (int)Math.Sqrt(input.Length);
            int kernelWeightIndexBase = index * Layer.KernelWidth * Layer.KernelWidth;

            for (int i = 0; i < Layer.FeatureMapWidth * Layer.FeatureMapWidth; i++)
            {
                Output[i] = 0;
            }

            Parallel.For(0, Layer.FeatureMapWidth, y =>
            {
                for (int x = 0; x < Layer.FeatureMapWidth; x++)
                {
                    int outputIndex = x + y * Layer.FeatureMapWidth;

                    for (int ky = 0; ky < Layer.KernelWidth; ky++)
                    {
                        for (int kx = 0; kx < Layer.KernelWidth; kx++)
                        {
                            int inputIndex = kx + x * Layer.StepSize + ky * inputFMWidth + y * (inputFMWidth * Layer.StepSize);
                            int weightIndex = kernelWeightIndexBase + kx + ky * Layer.KernelWidth;

                            Output[outputIndex] += input[inputIndex] * KernelWeights[weightIndex];
                        }
                    }
                }
            });
        }

        public void Backpropagate(double[] nextOutputs, int nextFeatureMapWidth, double[] fmOutputGradients, double learningRate, double momentum, int index, double weightDecay, MiniBatchMode miniBatchMode)
        {
            if (miniBatchMode == MiniBatchMode.Off || miniBatchMode == MiniBatchMode.First)
            {
                ResetWeightGradients();
            }

            int nextOutputWidth = (int)Math.Sqrt(nextOutputs.Length);

            Parallel.For(0, Layer.NumPrevFeatureMaps, k =>
            {
                if (Layer.FMIsConnectedToPrevFM(index, k))
                {
                    //pixel 0 of the next layer feature map associated with the current kernel
                    int nextNodeIndexFMBase = k * nextFeatureMapWidth * nextFeatureMapWidth;

                    for (int y = 0; y < Layer.FeatureMapWidth; y++)
                    {
                        for (int x = 0; x < Layer.FeatureMapWidth; x++)
                        {
                            //previously calculated error gradient of the current feature map pixel
                            double dErrY = fmOutputGradients[x + y * Layer.FeatureMapWidth];

                            //top left corner at the current kernel position in the next output
                            int nextNodeIndexBase = nextNodeIndexFMBase + Layer.StepSize * x + Layer.StepSize * nextFeatureMapWidth * y;

                            for (int ky = 0; ky < Layer.KernelWidth; ky++)
                            {
                                for (int kx = 0; kx < Layer.KernelWidth; kx++)
                                {
                                    KernelWeightGradents[k * Layer.KernelWidth * Layer.KernelWidth + kx + ky * Layer.KernelWidth] += dErrY * nextOutputs[nextNodeIndexBase + kx + ky * nextFeatureMapWidth];
                                }
                            }
                        }
                    }
                }
            });

            if (miniBatchMode == MiniBatchMode.Off || miniBatchMode == MiniBatchMode.Compute)
            {
                Parallel.For(0, Layer.NumPrevFeatureMaps, i =>
                {
                    if (Layer.FMIsConnectedToPrevFM(index, i))
                    {
                        for (int j = 0; j < Layer.KernelWidth * Layer.KernelWidth; j++)
                        {
                            //Weight decay
                            KernelWeightGradents[i * Layer.KernelWidth * Layer.KernelWidth + j] -= weightDecay * learningRate * KernelWeights[i * Layer.KernelWidth * Layer.KernelWidth + j];

                            KernelWeights[i * Layer.KernelWidth * Layer.KernelWidth + j] += learningRate * KernelWeightGradents[i * Layer.KernelWidth * Layer.KernelWidth + j];

                            //Add momentum
                            KernelWeights[i * Layer.KernelWidth * Layer.KernelWidth + j] += momentum * PrevKernelWeightGradients[i * Layer.KernelWidth * Layer.KernelWidth + j];
                            PrevKernelWeightGradients[i * Layer.KernelWidth * Layer.KernelWidth + j] = KernelWeightGradents[i * Layer.KernelWidth * Layer.KernelWidth + j];
                        }
                    }
                });
            }
        }

        public void RandomizeWeights(Random rand)
        {
            Bias = rand.NextDouble() * 2 - 1;

            for (int i = 0; i < Layer.NumPrevFeatureMaps * Layer.KernelWidth * Layer.KernelWidth; i++)
            {
                KernelWeights[i] = rand.NextDouble() * 2 - 1;
            }
        }

        private double[] CreateGaussianKernel(int width)
        {
            int centerX = width / 2;
            int centerY = width / 2;
            double spread = 0.5;
            double amplitude = -1;

            double[] data = new double[width * width];

            for (int y = 0; y < width; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    data[y * Layer.KernelWidth + x] = amplitude * Math.Exp(-(spread * (Math.Pow(x - centerX, 2)) + spread * (Math.Pow(y - centerY, 2))));
                }
            }

            return data;
        }
    }
}
