using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.DataSetProviders
{
    public class EvenOrOddDataSetProvider : IDataSetProvider
    {
        public static int InputSize_ = 7;

        public static List<TrainingExample> GetRandomExamples(int count)
        {
            List<TrainingExample> examples = new List<TrainingExample>();

            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < count; i++)
            {
                double[] input = new double[InputSize_];
                int cnt = 0;

                for (int j = 0; j < InputSize_; j++)
                {
                    if (rand.NextDouble() >= 0.5)
                    {
                        input[j] = 1;
                        cnt++;
                    }
                    else
                    {
                        input[j] = 0;
                    }
                }

                examples.Add(new TrainingExample(input, new double[1] { (cnt % 2 == 0 ? 1.0 : 0.0) }));
            }

            return examples;
        }

        public override List<TrainingExample> GetTrainingExamples()
        {
            return GetRandomExamples(1000);
        }

        public override List<TrainingExample> GetTestingExamples()
        {
            return GetRandomExamples(20);
        }

        public override int InputSize()
        {
            return EvenOrOddDataSetProvider.InputSize_;
        }

        public override int ResultSize()
        {
            return 1;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            return Math.Round(actual[0]) == expected[0];
        }
    }
}
