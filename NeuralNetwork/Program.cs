using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //SameOrDifferent();
            //LogicalAnd();
            //LogicalOr();
            //LogicalXOr();
            //EvenOrOdd();   //nope
            SinFunction();
        }

        static void SameOrDifferent()
        {
            ITrainingSetProvider trainingSetProvider = new SameOrDifferentTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        4,
                                                        ActivationFunctions.Tanh,
                                                        ActivationFunctions.Tanh,
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void LogicalAnd()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalAndTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        10,
                                                        ActivationFunctions.Sigmoid,
                                                        ActivationFunctions.Sigmoid,
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void LogicalOr()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalOrTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        10,
                                                        ActivationFunctions.Sigmoid,
                                                        ActivationFunctions.Sigmoid,
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void LogicalXOr()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalXOrTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        10,
                                                        ActivationFunctions.Sigmoid,
                                                        ActivationFunctions.Sigmoid,
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void EvenOrOdd()
        {
            ITrainingSetProvider trainingSetProvider = new EvenOrOddTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        20,
                                                        ActivationFunctions.Sigmoid,
                                                        ActivationFunctions.Sigmoid,
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(100000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void SinFunction()
        {
            ITrainingSetProvider trainingSetProvider = new SinFunctionTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        20,
                                                        ActivationFunctions.Sigmoid,
                                                        ActivationFunctions.Sigmoid,
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(10000000, false);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }
    }
}
