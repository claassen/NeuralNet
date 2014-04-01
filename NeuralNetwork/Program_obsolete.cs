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
            //EvenOrOdd();  
            //SinFunction();
            //MNIST();
        }

        /*static void SameOrDifferent()
        {
            ITrainingSetProvider trainingSetProvider = new SameOrDifferentTrainingSetProvider();

            NetworkManager manager = new NetworkManager(,
                                                        trainingSetProvider,
                                                        new InputLayer(2),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctionType.Tanh, 10)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Tanh, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000);

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
                                                            new HiddenLayer(ActivationFunctionType.Sigmoid, 10)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000);

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
                                                            new HiddenLayer(ActivationFunctionType.Sigmoid, 10)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000);

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
                                                            new HiddenLayer(ActivationFunctionType.Sigmoid, 10)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Sigmoid, 1),
                                                        0.05);

            Console.WriteLine("Training...");
            manager.TrainNetwork(1000000);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void EvenOrOdd()
        {
            ITrainingSetProvider trainingSetProvider = new EvenOrOddTrainingSetProvider();

           NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(7),
                                                        new List<HiddenLayer>()
                                                        {
                                                            //new ConvolutionalLayer(ActivationFunctionType.Sigmoid, 2, 3),
                                                            new HiddenLayer(ActivationFunctionType.Sigmoid, 50)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Sigmoid, 1),
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(10000000);

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
                                                            new HiddenLayer(ActivationFunctionType.Sigmoid, 20)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Sigmoid, 1),
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(10000000);

            manager.ShowNetwork();

            manager.TestNetwork();
        }

        static void MNIST()
        {
            ITrainingSetProvider trainingSetProvider = new MNISTTrainingSetProvider();

            NetworkManager manager = new NetworkManager(trainingSetProvider,
                                                        new InputLayer(28*28),
                                                        new List<HiddenLayer>()
                                                        {
                                                            new HiddenLayer(ActivationFunctionType.Tanh, 300)
                                                        },
                                                        new OutputLayer(ActivationFunctionType.Tanh, 10),
                                                        0.25);

            Console.WriteLine("Training...");
            manager.TrainNetwork(100000);

            //manager.ShowNetwork();

            manager.TestNetwork();
        }*/
    }
}
