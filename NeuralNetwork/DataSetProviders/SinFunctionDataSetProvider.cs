using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.DataSetProviders
{
    public class SinFunctionDataSetProvider : IDataSetProvider
    {
        public static List<TrainingExample> GetSinExamples(int count)
        {
            List<TrainingExample> examples = new List<TrainingExample>();

            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < count; i++)
            {
                double input = rand.NextDouble() * Math.PI * 4;
                examples.Add(new TrainingExample(new double[1] { input }, new double[1] { 0.5 * (Math.Sin(input) + 1) }));
            }

            return examples;
        }

        public override List<TrainingExample> GetTrainingExamples()
        {
            return GetSinExamples(1000);
        }

        public override List<TrainingExample> GetTestingExamples()
        {
            return GetSinExamples(100);
        }

        public override int InputSize()
        {
            return 1;
        }

        public override int ResultSize()
        {
            return 1;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            return Math.Round(expected[0], 2) == Math.Round(actual[0], 2);
        }
    }
}
