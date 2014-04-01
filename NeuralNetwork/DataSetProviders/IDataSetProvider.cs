using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Xml.Serialization;

namespace NeuralNetwork.DataSetProviders
{
    [Serializable]
    [XmlInclude(typeof(MNISTDataSetProvider)),
     XmlInclude(typeof(SinFunctionDataSetProvider)),
     XmlInclude(typeof(LogicalAndDataSetProvider)),
     XmlInclude(typeof(LogicalXOrDataSetProvider))]
    public abstract class IDataSetProvider
    {
        public abstract List<TrainingExample> GetTrainingExamples();
        public virtual List<TrainingExample> GetTestingExamples()
        {
            return GetTrainingExamples();
        }
        public abstract int InputSize();
        public abstract int ResultSize();
        public abstract bool IsCorrect(double[] expected, double[] actual);
        public virtual bool ScaleInput()
        {
            return false;
        }
        public virtual double InputMin()
        {
            return 0;
        }
        public virtual double InputMax()
        {
            return 0;
        }
        public virtual double ScaleMin()
        {
            return 0;
        }
        public virtual double ScaleMax()
        {
            return 0;
        }
    }
}
