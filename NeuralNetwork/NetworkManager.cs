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

        public NetworkManager(string networkName, ITrainingSetProvider trainingSetProvider, InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer, double learningRate = 0.25)
        {
            m_TrainingSetProvider = trainingSetProvider;
            m_TrainingExamples = trainingSetProvider.GetTrainingExamples();

            m_Network = new NeuralNetwork(networkName, inputLayer, hiddenLayers, outputLayer, learningRate);

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

        public void TrainNetwork(long iterations, bool loadPrevious, Func<NeuralNetwork, double, bool> onTrain = null)
        {
            bool useAdaptiveLearningRate = true;

            if (loadPrevious)
            {
                m_Network.LoadFromDisk("C:\\nets");
            }

            if (!useAdaptiveLearningRate)
            {
                for (long i = 0; i < iterations; i++)
                {
                    TrainingExample example = GetRandomTrainingExample();

                    double error = m_Network.Train(example.Input, example.Expected);

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
            double dE = 0;
            double dEAvg = 0;
            double prevError = TestNetwork(null, 100);
            double minError = prevError;
            int minErrorIter = 0;

            double lrInc = 1.1;
            double lrDec = 0.5;
            int maxItersBeforeRevertingToLastBest = 5000;
            double minErrorChange = 0.01;
            double alpha = 0.1;
            int numTrainingExamples = 10;

            m_Network.LearningRate = 0.0000000001;

            for (int i = 0; i < iterations; i++)
            {
                TrainingExample example = GetRandomTrainingExample();
                double tError = m_Network.Train(example.Input, example.Expected);

                double error = TestNetwork(null, numTrainingExamples) / numTrainingExamples;//m_Network.Train(example.Input, example.Expected);

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

            m_Network.LoadFromDisk("C:\\nets\\bestsofar.xml");
        }

        public void ShowNetwork()
        {
            m_Network.DisplayNetwork();
        }

        /// <summary>
        /// Tests the network using the current provider and returns the total error.
        /// </summary>
        /// <param name="onTest">Callback function to be called on each test.
        ///                      Parameters: input, actual, expected</param>
        /// <returns>Total error over all testing examples.</returns>
        public double TestNetwork(Func<double[], double[], double[], bool> onTest = null, int maxIterations = 0)
        {
            List<TrainingExample> examples = m_TrainingSetProvider.GetTestingExamples();
            
            double error = 0;
            
            for (int i = 0; i < examples.Count; i++)
            {
                if (maxIterations > 0 && i >= maxIterations)
                {
                    break;
                }

                TrainingExample example = examples[i];

                double[] result = m_Network.GetResult(example.Input);

                for (int j = 0; j < result.Length; j++)
                {
                    error += Math.Abs(example.Expected[j] - result[j]);
                }

                if (null != onTest)
                {
                    if (!onTest(example.Input, result, example.Expected))
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
