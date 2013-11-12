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
                                                        new InputLayer(2),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Tanh, 10)
                                                        },
                                                        new OutputLayer(ActivationFunctions.Tanh, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }
        
        static void LogicalAnd()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalAndTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(2),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Sigmoid, 10)
                                                        }, 
                                                        new OutputLayer(ActivationFunctions.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void LogicalOr()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalOrTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(2),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Sigmoid, 10)
                                                        }, 
                                                        new OutputLayer(ActivationFunctions.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void LogicalXOr()
        {
            ITrainingSetProvider trainingSetProvider = new LogicalXOrTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(2),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Sigmoid, 10)
                                                        }, 
                                                        new OutputLayer(ActivationFunctions.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void EvenOrOdd()
        {
            ITrainingSetProvider trainingSetProvider = new EvenOrOddTrainingSetProvider();

           NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(4),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Sigmoid, 50)
                                                        }, 
                                                        new OutputLayer(ActivationFunctions.Sigmoid, 1),
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(10000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void SinFunction()
        {
            ITrainingSetProvider trainingSetProvider = new SinFunctionTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(1),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctions.Sigmoid, 20)
                                                        }, 
                                                        new OutputLayer(ActivationFunctions.Sigmoid, 1),
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(10000000, false);

            manager.ShowNetwork();

            manager.TestNetwork();
        }
        
    }
}
