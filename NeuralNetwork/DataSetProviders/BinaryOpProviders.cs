using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.DataSetProviders
{
    public class SameOrDifferentDataSetProvider : IDataSetProvider
    {
        public override List<TrainingExample> GetTrainingExamples()
        {
            return new List<TrainingExample>()
            {
                new TrainingExample(new double[] { 1.0, 1.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 1.0, 0.0 }, new double[] { 0.0 }),
                new TrainingExample(new double[] { 0.0, 1.0 }, new double[] { 0.0 }),
                new TrainingExample(new double[] { 0.0, 0.0 }, new double[] { 1.0 })
            };
        }

        public override int InputSize()
        {
            return 2;
        }

        public override int ResultSize()
        {
            return 1;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            throw new NotImplementedException();
        }
    }

    public class LogicalAndDataSetProvider : IDataSetProvider
    {
        public override List<TrainingExample> GetTrainingExamples()
        {
            return new List<TrainingExample>()
            {
                new TrainingExample(new double[] { 1.0, 1.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 1.0, 0.0 }, new double[] { 0.0 }),
                new TrainingExample(new double[] { 0.0, 1.0 }, new double[] { 0.0 }),
                new TrainingExample(new double[] { 0.0, 0.0 }, new double[] { 0.0 })
            };
        }

        public override int InputSize()
        {
            return 2;
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

    public class LogicalOrDataSetProvider : IDataSetProvider
    {
        public override List<TrainingExample> GetTrainingExamples()
        {
            return new List<TrainingExample>()
            {
                new TrainingExample(new double[] { 1.0, 1.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 1.0, 0.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 0.0, 1.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 0.0, 0.0 }, new double[] { 0.0 })
            };
        }

        public override int InputSize()
        {
            return 2;
        }

        public override int ResultSize()
        {
            return 1;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            throw new NotImplementedException();
        }
    }

    public class LogicalXOrDataSetProvider : IDataSetProvider
    {
        public override List<TrainingExample> GetTrainingExamples()
        {
            return new List<TrainingExample>()
            {
                new TrainingExample(new double[] { 1.0, 1.0 }, new double[] { 0.0 }),
                new TrainingExample(new double[] { 1.0, 0.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 0.0, 1.0 }, new double[] { 1.0 }),
                new TrainingExample(new double[] { 0.0, 0.0 }, new double[] { 0.0 })
            };
        }

        public override int InputSize()
        {
            return 2;
        }

        public override int ResultSize()
        {
            return 1;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            return (expected[0] < 0.5 && actual[0] < 0.5) || (expected[0] > 0.5 && actual[0] > 0.5);
        }
    }
}
