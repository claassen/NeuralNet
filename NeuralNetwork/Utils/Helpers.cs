using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace NeuralNetwork.Utils
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

        public static Image ImageFromDoubleArray(double[] data, int desiredWidth = 0)
        {
            int origWidth = (int)Math.Sqrt(data.Length);

            byte[] byteData = new byte[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                byteData[i] = (byte)data[i];
            }

            Bitmap bmp = new Bitmap(origWidth, origWidth);

            for (int i = 0; i < origWidth; i++)
            {
                for (int j = 0; j < origWidth; j++)
                {
                    byte b = byteData[i * origWidth + j];
                    Color c = Color.FromArgb(255, b, b, b);

                    bmp.SetPixel(j, i, c);
                }
            }

            int newWidth = (desiredWidth == 0 ? origWidth * 3 : desiredWidth);

            return ResizeBitmap(bmp, newWidth, newWidth);
        }

        public static Bitmap ResizeBitmap(Bitmap sourceBMP, int width, int height)
        {
            Bitmap result = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format16bppRgb555);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.DrawImage(sourceBMP, 0, 0, width, height);
            }
            sourceBMP.Dispose();
            return result;
        }
    }
}
