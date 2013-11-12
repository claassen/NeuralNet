using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public abstract class ITrainingSetProvider
    {
        public abstract List<TrainingExample> GetTrainingExamples();
        public virtual List<TrainingExample> GetTestingExamples()
        {
            return GetTrainingExamples();
        }
        public abstract int InputSize();
        public abstract int ResultSize();
    }

    public class SameOrDifferentTrainingSetProvider : ITrainingSetProvider
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
    }

    public class LogicalAndTrainingSetProvider : ITrainingSetProvider
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
    }

    public class LogicalOrTrainingSetProvider : ITrainingSetProvider
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
    }

    public class LogicalXOrTrainingSetProvider : ITrainingSetProvider
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
    }

    public class EvenOrOddTrainingSetProvider : ITrainingSetProvider
    {
        public static int InputSize_ = 4;

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
            return EvenOrOddTrainingSetProvider.InputSize_;
        }

        public override int ResultSize()
        {
            return 1;
        }
    }

    /*
     * Good settings: 
     *  10000000 iterations, learning rate 0.25, 1000 training sets, 20 hidden nodes, sigmoid activation functions
     */
    public class SinFunctionTrainingSetProvider : ITrainingSetProvider
    {
        public static List<TrainingExample> GetSinExamples(int count)
        {
            List<TrainingExample> examples = new List<TrainingExample>();

            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < count; i++)
            {
                double input = rand.NextDouble() * Math.PI * 4;
                examples.Add(new TrainingExample(new double[1] { input }, new double[1] { 0.5*(Math.Sin(input) + 1) }));
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
    }
}
