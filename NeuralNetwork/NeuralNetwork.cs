using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private ActivationFunction m_OutputActivationFunction;
        private ActivationFunction m_HiddenActivationFunction;

        private double m_LearningRate;
        private Random m_Rand;

        private int m_NumInput;
        private int m_NumOutput;
        private int m_NumHidden;

        private int m_NumOutputWeights;
        private int m_NumHiddenWeights;

        private double[,] m_HiddenWeights;
        private double[,] m_OutputWeights;

        private double[] m_Inputs;
        private double[] m_HiddenOutputs;

        private double[] m_OutputOutputs;

        private double[] m_OutputGradients;
        private double[] m_HiddenGradients;

        private double[,] m_OutputWeightGradients;
        private double[,] m_HiddenWeightGradients;

        private const double MOMENTUM = 0.4;
        private double[,] m_PrevOutputWeightGradients;
        private double[,] m_PrevHiddenWeightGradients;

        public NeuralNetwork(int numIn, int numOut, int numHidden, ActivationFunction outputActivationFunction, ActivationFunction hiddenActivationFunction, double learningRate = 0.25)
        {
            m_LearningRate = learningRate;

            m_NumInput = numIn;
            m_NumOutput = numOut;
            m_NumHidden = numHidden;

            m_NumOutputWeights = m_NumHidden + 1;
            m_NumHiddenWeights = m_NumInput + 1;

            m_HiddenWeights = new double[m_NumHidden, m_NumHiddenWeights];
            m_OutputWeights = new double[m_NumOutput, m_NumOutputWeights];

            m_Inputs = new double[m_NumInput];
            m_HiddenOutputs = new double[m_NumHidden];
            m_OutputOutputs = new double[m_NumOutput];

            m_OutputGradients = new double[m_NumOutput];
            m_HiddenGradients = new double[m_NumHidden];

            m_OutputWeightGradients = new double[m_NumOutput, m_NumOutputWeights];
            m_HiddenWeightGradients = new double[m_NumHidden, m_NumHiddenWeights];

            //Activation functions
            m_OutputActivationFunction = outputActivationFunction;
            m_HiddenActivationFunction = hiddenActivationFunction;

            //Momentum
            m_PrevOutputWeightGradients = new double[m_NumOutput, m_NumOutputWeights];
            m_PrevHiddenWeightGradients = new double[m_NumHidden, m_NumHiddenWeights];

            //Seed random number generator
            m_Rand = new Random(DateTime.Now.Millisecond);

            RandomizeWeights();
        }

        public void RandomizeWeights()
        {
            for (int i = 0; i < m_NumHidden; i++)
            {
                for (int j = 0; j < m_NumHiddenWeights; j++)
                {
                    m_HiddenWeights[i, j] = m_Rand.NextDouble() * 2 - 1;
                }
            }

            for (int i = 0; i < m_NumOutput; i++)
            {
                for (int j = 0; j < m_NumOutputWeights; j++)
                {
                    m_OutputWeights[i, j] = m_Rand.NextDouble() * 2 - 1;
                }
            }
        }

        public double[] GetResult(double[] input)
        {
            if (input.Length != m_NumInput)
                throw new ArgumentException("Input length does not match network input size");

            //Hidden nodes
            for (int i = 0; i < m_NumHidden; i++)
            {
                //Hidden node i
                double sum = 0;

                for (int j = 0; j < m_NumHiddenWeights; j++)
                {
                    //Input j
                    sum += m_HiddenWeights[i, j] * (j == 0 ? 1 : input[j-1]);
                }
                m_HiddenOutputs[i] = m_HiddenActivationFunction.Function(sum);
            }

            //Output nodes
            for (int i = 0; i < m_NumOutput; i++)
            {
                //Output node i
                double sum = 0;

                for (int j = 0; j < m_NumOutputWeights; j++)
                {
                    //Hidden output j
                    sum += m_OutputWeights[i, j] * (j == 0 ? 1 : m_HiddenOutputs[j-1]);
                }
                m_OutputOutputs[i] = m_OutputActivationFunction.Function(sum);
            }

            return m_OutputOutputs;
        }

        /*
         * Trains the neural network using the given training example. ("Online" training)
         * 
         * Variable names in loops follow the convention that they represent the following layers:
         * k = output layer
         * j = hidden layer
         * i = input layer
         */
        public void Train(double[] input, double[] expected)
        {
            double[] actual = GetResult(input);

            //Calculate output and output weight gradients
            for (int k = 0; k < m_NumOutput; k++)
            {
                m_OutputGradients[k] = (expected[k] - actual[k]) * m_OutputActivationFunction.Derivative(actual[k]);
                
                for (int j = 0; j < m_NumOutputWeights; j++)
                {
                    m_OutputWeightGradients[k, j] = m_LearningRate * m_OutputGradients[k] * (j > 0 ? m_HiddenOutputs[j - 1] : 1);

                    //Update weight
                    m_OutputWeights[k, j] += m_OutputWeightGradients[k, j];

                    //Add momentum
                    m_OutputWeights[k, j] += MOMENTUM * m_PrevOutputWeightGradients[k, j];

                    m_PrevOutputWeightGradients[k, j] = m_OutputWeightGradients[k, j];
                }
            }

            //Calculate hidden and hidden weight gradients
            for (int j = 0; j < m_NumHidden; j++)
            {
                //The sum of the output gradients (calculated above) of each output node connected to the current hidden node
                //times the output of the current node
                double outputGradientSum = 0;
                for(int k = 0; k < m_NumOutput; k++)
                {
                    outputGradientSum += m_OutputGradients[k] * m_HiddenOutputs[j];
                }

                //Gradient for this hidden node is the sum calculated above times the derivative of the activation
                //function at the output of the current node
                m_HiddenGradients[j] = m_HiddenActivationFunction.Derivative(m_HiddenOutputs[j]) * outputGradientSum;

                for (int i = 0; i < m_NumHiddenWeights; i++)
                {
                    m_HiddenWeightGradients[j, i] = m_LearningRate * m_HiddenGradients[j] * (i > 0 ? m_Inputs[i-1] : 1);

                    //Update weight
                    m_HiddenWeights[j, i] += m_HiddenWeightGradients[j, i];

                    //Add momentum
                    m_HiddenWeights[j, i] += MOMENTUM * m_PrevHiddenWeightGradients[j, i];

                    m_PrevHiddenWeightGradients[j, i] = m_HiddenWeightGradients[j, i];
                }
            }
        }

        public void PrettyDisplay()
        {
            Console.WriteLine("\n::OUTPUT LAYER::\n");
            for (int i = 0; i < m_NumOutput; i++)
            {
                Console.WriteLine("OUTPUT NODE " + i + 1);
                Console.WriteLine("OUTPUT: " + m_OutputOutputs[i]);
                Console.Write("WEIGHTS: ");
                for (int j = 0; j < m_NumOutputWeights; j++)
                {
                    Console.Write(m_OutputWeights[i,j] + " ");
                }
                Console.Write("\nGRADIENTS: ");
                for (int j = 0; j < m_NumOutputWeights; j++)
                {
                    Console.Write(m_OutputWeightGradients[i,j] + " ");
                }
                Console.WriteLine("\n");
            }

            Console.WriteLine("\n::HIDDEN LAYER::\n");
            for (int i = 0; i < m_NumHidden; i++)
            {
                Console.WriteLine("HIDDEN NODE " + i + 1);
                Console.WriteLine("OUTPUT: " + m_HiddenOutputs[i]);
                Console.Write("WEIGHTS: ");
                for (int j = 0; j < m_NumHiddenWeights; j++)
                {
                    Console.Write(m_HiddenWeights[i,j] + " ");
                }
                Console.Write("\nGRADIENTS: ");
                for (int j = 0; j < m_NumHiddenWeights; j++)
                {
                    Console.Write(m_HiddenWeightGradients[i, j] + " ");
                }
                Console.WriteLine("\n");
            }
        }
    }
}
