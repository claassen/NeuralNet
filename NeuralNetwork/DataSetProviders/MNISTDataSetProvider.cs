using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using NeuralNetwork.Utils;

namespace NeuralNetwork.DataSetProviders
{
    public class MNISTDataSetProvider : IDataSetProvider
    {
        private const string MNIST_SOURCE_PATH = "C:\\MNIST";

        private List<TrainingExample> GetMNISTExamples(string labelFile, string imageFile, int max = 0)
        {
            /*
             * MNIST file format: http://yann.lecun.com/exdb/mnist/
             */
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
            int numRows = Helpers.Int32FromBigEndianByteArray(imageBr.ReadBytes(4)) + 1;
            int numCols = Helpers.Int32FromBigEndianByteArray(imageBr.ReadBytes(4)) + 1;

            for (int i = 0; i < (max != 0 ? Math.Min(max, numImages) : numImages); i++)
            {
                double[] imageData = new double[numRows * numCols];
                double[] expected = new double[10];

                for (int r = 0; r < numRows; r++)
                {
                    for (int c = 0; c < numCols; c++)
                    {
                        if (c < (numCols - 1) && r < (numRows - 1))
                        {
                            imageData[r * numCols + c] = imageBr.ReadByte();
                        }
                        else
                        {
                            imageData[r * numCols + c] = 0;
                        }
                    }
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
            return 29 * 29;
        }

        public override int ResultSize()
        {
            return 10;
        }

        public override bool IsCorrect(double[] expected, double[] actual)
        {
            return Helpers.GetMaxValueIndex(actual) == Helpers.GetMaxValueIndex(expected);
        }

        public override bool ScaleInput()
        {
            return true;
        }

        public override double InputMin()
        {
            return 0;
        }

        public override double InputMax()
        {
            return 255;
        }

        public override double ScaleMin()
        {
            return 0;
        }

        public override double ScaleMax()
        {
            return 1;
        }
    }
}
