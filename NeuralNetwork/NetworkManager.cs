using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NetworkManager
    {
        private ITrainingSetProvider m_TrainingSetProvider;
        private List<TrainingExample> m_TrainingExamples;
        private NeuralNetwork m_Network;
        private Random m_Rand;

        public NetworkManager(ITrainingSetProvider trainingSetProvider, InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer, double learningRate = 0.25)
        {
            m_TrainingSetProvider = trainingSetProvider;
            m_TrainingExamples = trainingSetProvider.GetTrainingExamples();

            m_Network = new NeuralNetwork(inputLayer, hiddenLayers, outputLayer, learningRate);

            m_Rand = new Random(DateTime.Now.Millisecond);
        }

        public TrainingExample GetRandomTrainingExample()
        {
            return m_TrainingExamples[m_Rand.Next(m_TrainingExamples.Count)];
        }

        public void TrainNetwork(long iterations, bool interactive = false)
        {
            for (long i = 0; i < iterations; i++)
            {
                TrainingExample example = GetRandomTrainingExample();
                m_Network.Train(example.Input, example.Expected);

                if (interactive)
                {
                    m_Network.DisplayNetwork();
                    Console.ReadLine();
                }
            }
        }

        public void ShowNetwork()
        {
            m_Network.DisplayNetwork();
        }

        public void TestNetwork()
        {
            List<TrainingExample> examples = m_TrainingSetProvider.GetTestingExamples();

            double error = 0;

            for (int i = 0; i < examples.Count; i++)
            {
                TrainingExample example = examples[i];

                double[] result = m_Network.GetResult(example.Input);

                error *= i;
                for (int j = 0; j < example.Expected.Length; j++)
                {
                    error += Math.Abs(example.Expected[j] - result[j]);
                }
                error /= i + 1;

                Console.Write("Result for: [");
                for (int j = 0; j < example.Input.Length; j++)
                {
                    Console.Write((j > 0 ? ", " : "") + example.Input[j]);
                }
                Console.WriteLine("]");

                for (int j = 0; j < example.Expected.Length; j++)
                {
                    Console.WriteLine(Math.Round(result[j], 3, MidpointRounding.AwayFromZero));
                }

                Console.WriteLine("Expected: ");
                for (int j = 0; j < example.Expected.Length; j++)
                {
                    Console.WriteLine(example.Expected[j] + " ");
                }
                Console.WriteLine();
            }

            Console.WriteLine("AVERAGE ERROR: " + error);
        }
    }
}
