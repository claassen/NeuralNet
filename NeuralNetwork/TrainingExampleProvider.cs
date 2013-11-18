using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

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
            return EvenOrOddTrainingSetProvider.InputSize_;
        }

        public override int ResultSize()
        {
            return 1;
        }
    }

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

    /*
     * MNIST file format: http://yann.lecun.com/exdb/mnist/
     */
    public class MNISTTrainingSetProvider : ITrainingSetProvider
    {
        private const string MNIST_SOURCE_PATH = "C:\\MNIST";

        private List<TrainingExample> GetMNISTExamples(string labelFile, string imageFile, int max = 0)
        {
            BinaryReader labelBr = new BinaryReader(new FileStream(labelFile, FileMode.Open));
            BinaryReader imageBr = new BinaryReader(new FileStream(imageFile, FileMode.Open));

            List<TrainingExample> examples = new List<TrainingExample>();

            //Read header (32 bits)
            labelBr.ReadInt32();

            int numImages = Helpers.Int32FromBigEndianByteArray(labelBr.ReadBytes(4));

            //Read header (32 bits)
            imageBr.ReadInt32();

            //Image file also contains number of images, just check to make sure its the same one we already read from the label file
            if (numImages != Helpers.Int32FromBigEndianByteArray(imageBr.ReadBytes(4)))
            {
                labelBr.Close();
                imageBr.Close();
                throw new InvalidDataException("Number of images in label file does not match number of images in image file.");
            }

            //Image size
            int numRows = Helpers.Int32FromBigEndianByteArray(imageBr.ReadBytes(4));
            int numCols = Helpers.Int32FromBigEndianByteArray(imageBr.ReadBytes(4));

            for (int i = 0; i < (max != 0 ? Math.Min(max, numImages) : numImages); i++)
            {
                double[] imageData = new double[numRows * numCols];
                double[] expected = new double[10];

                for (int p = 0; p < numRows * numCols; p++)
                {
                    imageData[p] = imageBr.ReadByte();
                }

                expected[labelBr.ReadByte()] = 1.0;

                examples.Add(new TrainingExample(imageData, expected));
            }

            labelBr.Close();
            imageBr.Close();

            return examples;
        }

        public override List<TrainingExample> GetTrainingExamples()
        {
            return GetMNISTExamples(Path.Combine(MNIST_SOURCE_PATH, "train-labels.idx1-ubyte"), 
                                    Path.Combine(MNIST_SOURCE_PATH, "train-images.idx3-ubyte"));
        }

        public override List<TrainingExample> GetTestingExamples()
        {
            return GetMNISTExamples(Path.Combine(MNIST_SOURCE_PATH, "t10k-labels.idx1-ubyte"),
                                    Path.Combine(MNIST_SOURCE_PATH, "t10k-images.idx3-ubyte"));
        }

        public override int InputSize()
        {
            return 28 * 28;
        }

        public override int ResultSize()
        {
            return 10;
        }
    }
}
