using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public static class Helpers
    {
        public static int Int32FromBigEndianByteArray(byte[] data)
        {
            Array.Reverse(data);
            return BitConverter.ToInt32(data, 0);
        }

        public static int GetMaxValueIndex(double[] array)
        {
            double max = 0;
            int index = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] > max)
                {
                    max = array[i];
                    index = i;
                }
            }

            return index;
        }
    }
}
