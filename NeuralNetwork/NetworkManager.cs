using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.DataSetProviders;
using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    public class NetworkManager
    {
        private List<TrainingExample> m_TrainingExamples;
        private List<TrainingExample> m_TestingExamples;
        private NeuralNetwork m_Network;
        private Random m_Rand;

        public NetworkManager(NeuralNetwork network, LearningMethod learningMethod, double learningRate, double momentum, double weightDecay)
        {
            m_TrainingExamples = network.InputLayer.DataSetProvider.GetTrainingExamples();
            m_TestingExamples = network.InputLayer.DataSetProvider.GetTestingExamples();

            m_Network = network;

            m_Network.LearningRate = learningRate;
            m_Network.Momentum = momentum;
            m_Network.WeightDecay = weightDecay;
            m_Network.LearningMethod = learningMethod;

            m_Rand = new Random(DateTime.Now.Millisecond);
        }

        public NeuralNetwork GetNetwork()
        {
            return m_Network;
        }

        public TrainingExample GetRandomTrainingExample()
        {
            return m_TrainingExamples[m_Rand.Next(m_TrainingExamples.Count)];
        }

        public TrainingExample GetRandomTestingExample()
        {
            return m_TestingExamples[m_Rand.Next(m_TestingExamples.Count)];
        }

        public void TrainNetwork(long iterations, bool useAdaptiveLearningRate, bool useMiniBatch, int miniBatchSize, Func<NeuralNetwork, double, bool> onTrain = null)
        {
            if (!useAdaptiveLearningRate)
            {
                for (long i = 0; i < iterations; i++)
                {
                    double error;

                    if (useMiniBatch)
                    {
                        error = TrainMiniBatch(miniBatchSize);
                    }
                    else
                    {
                        error = m_Network.Train(GetRandomTrainingExample());
                    }

                    if (null != onTrain)
                    {
                        if (!onTrain(m_Network, error))
                        {
                            return;
                        }
                    }
                }

                return;
            }

            //Adaptive learning rate method.
            //This method was taken from this paper: http://lmb.informatik.uni-freiburg.de/papers/download/du_diss.pdf
            int numTrainingExamples = 20;
            double dE = 0;
            double dEAvg = 0;
            double prevError = TestNetwork(null, numTrainingExamples) / numTrainingExamples;
            double minError = prevError;
            int minErrorIter = 0;

            double lrInc = 1.1;
            double lrDec = 0.5;
            int maxItersBeforeRevertingToLastBest = 5000;
            double minErrorChange = 0.01;
            double alpha = 0.1;
            
            m_Network.LearningRate = 0.0000000001;

            for (int i = 0; i < iterations; i++)
            {
                double tError = 0;

                if (useMiniBatch)
                {
                    tError = TrainMiniBatch(miniBatchSize);
                }
                else
                {
                    tError = m_Network.Train(GetRandomTrainingExample());
                }
                
                double error = TestNetwork(null, numTrainingExamples) / numTrainingExamples;

                dE = (error - prevError) / error;
                prevError = error;

                if (dE * dEAvg < 0 && Math.Abs(dEAvg) > minErrorChange)
                {
                    //If the average was increasing error and we just decreased it or
                    //the average was decreasing error and we just increased it,
                    //and the avg change is greater than a threshold, then
                    //decrease the learning rate
                    m_Network.LearningRate *= lrDec;
                }
                else
                {
                    //If the average change is very small (in either direction),
                    //or if the the current change is going in the same direction as the
                    //average (either increasing or decreasing error) then
                    //increase the learning rate
                    m_Network.LearningRate *= lrInc;
                }

                dEAvg = alpha * dE + (1 - alpha) * dEAvg;

                if (error < minError)
                {
                    minError = error;
                    minErrorIter = i;
                    m_Network.SaveToDisk("C:\\nets");
                }

                if (i - minErrorIter > maxItersBeforeRevertingToLastBest)
                {
                    //load
                    minErrorIter = i;
                    m_Network.LoadFromDisk("C:\\nets");
                }

                if (null != onTrain)
                {
                    if (!onTrain(m_Network, error))
                    {
                        m_Network.LoadFromDisk("C:\\nets");
                        error = TestNetwork(null, numTrainingExamples) / numTrainingExamples;
                        onTrain(m_Network, error);
                        return;
                    }
                }
            }

            m_Network.LoadFromDisk("C:\\nets");
        }

        private double TrainMiniBatch(int batchSize)
        {
            m_Network.Train(GetRandomTrainingExample(), MiniBatchMode.First);

            for (int i = 1; i < batchSize; i++)
            {
                m_Network.Train(GetRandomTrainingExample(), MiniBatchMode.Accumulate);
            }

            return m_Network.Train(GetRandomTrainingExample(), MiniBatchMode.Compute);
        }

        public double TestNetwork(Func<NeuralNetwork, double[], double[], double[], bool> onTest = null, int maxIterations = 0)
        {
            double error = 0;
            
            for (int i = 0; i < m_TestingExamples.Count; i++)
            {
                if (maxIterations > 0 && i >= maxIterations)
                {
                    break;
                }

                TrainingExample example = m_TestingExamples[i];

                double[] result = m_Network.GetResult(example.Input);

                for (int j = 0; j < result.Length; j++)
                {
                    error += Math.Abs(example.Expected[j] - result[j]);
                }

                if (null != onTest)
                {
                    if (!onTest(m_Network, example.Input, result, example.Expected))
                    {
                        //onTest returns false to indicate testing should be stopped
                        return error;
                    }
                }
            }

            return error;
        }
    }
}
