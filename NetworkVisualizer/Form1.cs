using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using NeuralNetwork;
using System.Runtime.InteropServices;
using System.Threading;

namespace NetworkVisualizer
{
    public partial class MainWindow : Form
    {
        private NetworkManager m_Manager;
        private BackgroundWorker m_Worker;

        private FlowLayoutPanel[] layerPanels;
        private PictureBox[][] featureMapImages;

        private ITrainingSetProvider provider;
        private List<TrainingExample> testingExamples;
        private TrainingExample testExample;

        private int textExampleIndex = 0;

        private ManualResetEvent evt = new ManualResetEvent(false);
       
        public MainWindow()
        {
            InitializeComponent();
            label2.Visible = false;
            progressBar1.Visible = false;

            Init();
        }

        private void Init()
        {
            StopButton.Enabled = false;

            ITrainingSetProvider trainingSetProvider = new MNISTTrainingSetProvider();
            provider = trainingSetProvider;

            testingExamples = trainingSetProvider.GetTestingExamples();

            testExample = testingExamples[textExampleIndex];

            m_Manager = new NetworkManager("testnet4",
                                           trainingSetProvider,
                                           new InputLayer(28 * 28),
                                           new List<HiddenLayer>()
                                           {
                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 20, 1), 
                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 4, 60, 2), 
                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 120, 2),
                                               //new HiddenLayer(ActivationFunctionType.Tanh, 300)

                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 2, 10, 2), 
                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 60, 2), 
                                               //new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 120, 1),
                                               new HiddenLayer(ActivationFunctionType.Tanh, 800)
                                           },
                                           new OutputLayer(ActivationFunctionType.Softmax, 10),
                                           0.0005);


            SetupDisplay();
        }

        private void TrainButton_Click(object sender, EventArgs e)
        {
            StopButton.Enabled = true;
            TrainButton.Enabled = false;
            TestButton.Enabled = false;

            label2.Visible = true;
            progressBar1.Visible = true;

            TrainNetwork(0.01, false, chkDynamicLearningRate.Checked, chkUseMiniBatch.Checked);
        }

        private void TestButton_Click(object sender, EventArgs e)
        {
            TestNetwork();
        }

        private void StopButton_Click(object sender, EventArgs e)
        {
            m_Worker.CancelAsync();
        }

        private void TrainNetwork(double errorThreshold, bool loadPrev, bool useAdaptiveLearningRate, bool useMiniBatch)
        {
            m_Worker = new BackgroundWorker();
            m_Worker.WorkerSupportsCancellation = true;
            m_Worker.WorkerReportsProgress = true;

            int trainingIters = 120000;

            double avgError = 5;
            int epoch = 0;

            m_Worker.DoWork += new DoWorkEventHandler((o, args) =>
            {
                m_Manager.TrainNetwork(trainingIters, loadPrev, useAdaptiveLearningRate, useMiniBatch, (network, error) =>
                {
                    if (m_Worker.CancellationPending)
                    {
                        m_Worker.Dispose();
                        return false;
                    }

                    avgError *= 24;
                    avgError += error;
                    avgError /= 25;

                    epoch++;

                    if (epoch % 10 == 0 || avgError <= errorThreshold)
                    {
                        lblAvgError.BeginInvoke(new Action(() =>
                        {
                            lblAvgError.Text = "" + avgError;
                        }));

                        UpdateDisplay(testExample.Input, "");
                    }

                    if (epoch % 10 == 0)
                    {
                        m_Worker.ReportProgress((int)(((double)epoch / (double)trainingIters) * 100));
                    }

                    if (avgError <= errorThreshold) return false;

                    return true;
                });
            });

            m_Worker.ProgressChanged += new ProgressChangedEventHandler((o, args) =>
            {
                progressBar1.Value = (int)(args.ProgressPercentage);
            });

            m_Worker.RunWorkerCompleted += new RunWorkerCompletedEventHandler((o, args) =>
            {
                label2.Visible = false;
                progressBar1.Visible = false;
                StopButton.Enabled = false;
                TrainButton.Enabled = true;
                TestButton.Enabled = true;
            });

            m_Worker.RunWorkerAsync();
        }

        private void TestNetwork()
        {
            StopButton.Enabled = true;
            TrainButton.Enabled = false;
            TestButton.Enabled = false;

            int totalTests = 0;
            int totalCorrect = 0;

            m_Worker = new BackgroundWorker();
            m_Worker.WorkerSupportsCancellation = true;
            m_Worker.WorkerReportsProgress = true;

            int numIters = provider.GetTestingExamples().Count();
            int epoch = 0;

            DateTime last = DateTime.Now;

            m_Worker.DoWork += new DoWorkEventHandler((o, args) =>
            {
                try
                {
                    m_Manager.TestNetwork((double[] input, double[] result, double[] expected) =>
                    {
                        if (m_Worker.CancellationPending)
                        {
                            m_Worker.Dispose();
                            return false;
                        }

                        int guess = Helpers.GetMaxValueIndex(result);
                       
                        int expectedNum = Helpers.GetMaxValueIndex(expected);
                        
                        totalTests++;

                        if (expectedNum == guess)
                        {
                            totalCorrect++;
                        }
                        
                        epoch++;

                        if (epoch % 10 == 0)
                        {
                            double temp = ((double)epoch / (double)numIters) * 100;
                            m_Worker.ReportProgress((int)temp);
                        }

                        DateTime now = DateTime.Now;

                        if (now.Subtract(last).Seconds > 1)
                        {
                            last = now;
                            UpdateDisplay(input, guess.ToString());
                        }

                        return true;
                    });
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            });

            m_Worker.ProgressChanged += new ProgressChangedEventHandler((o, args) =>
            {
                progressBar1.Value = (int)(args.ProgressPercentage);
            });

            m_Worker.RunWorkerCompleted += new RunWorkerCompletedEventHandler((o, args) =>
            {
                StopButton.Enabled = false;
                TrainButton.Enabled = true;
                TestButton.Enabled = true;

                MessageBox.Show(totalCorrect + " out of " + totalTests + " correct");
            });

            label2.Visible = true;
            label2.Text = "Testing...";
            progressBar1.Visible = true;
            progressBar1.Value = 0;

            m_Worker.RunWorkerAsync();
        }

        private void SetupDisplay()
        {
            List<ConvolutionalLayer> convLayers = m_Manager.GetNetwork().HiddenLayers.Where(l => l is ConvolutionalLayer).Select(l => (ConvolutionalLayer)l).ToList();

            layerPanels = new FlowLayoutPanel[convLayers.Count];
            featureMapImages = new PictureBox[convLayers.Count][];

            for (int i = 0; i < layerPanels.Length; i++)
            {
                featureMapImages[i] = new PictureBox[convLayers[i].NumFeatureMaps];

                layerPanels[i] = new FlowLayoutPanel();
                layerPanels[i].FlowDirection = FlowDirection.TopDown;
                layerPanels[i].BorderStyle = BorderStyle.Fixed3D;
                layerPanels[i].Width = (flowLayoutPanel.Width / layerPanels.Length) - 20;
                layerPanels[i].Height = flowLayoutPanel.Height - 20;
                layerPanels[i].AutoScroll = true;

                for (int j = 0; j < convLayers[i].NumFeatureMaps; j++)
                {
                    PictureBox box = new PictureBox();
                    box.SizeMode = PictureBoxSizeMode.AutoSize;
                    
                    featureMapImages[i][j] = box;
                    layerPanels[i].Controls.Add(box);
                }

                flowLayoutPanel.Controls.Add(layerPanels[i]);
            }

            pictureBox1.Image = Helpers.ImageFromDoubleArray(testExample.Input, 48, 48);
            UpdateDisplay(testExample.Input, "");
        }

        private void UpdateDisplay(double[] input, string guess)
        {
            lock (this)
            {
                List<ConvolutionalLayer> convLayers = m_Manager.GetNetwork().HiddenLayers.Where(l => l is ConvolutionalLayer).Select(l => (ConvolutionalLayer)l).ToList();

                m_Manager.GetNetwork().GetResult(input);

                evt.Reset();

                Action uiThreadWork = new Action(() =>
                {
                    pictureBox1.Image = Helpers.ImageFromDoubleArray(input, 28, 28);
                    label1.Text = guess;

                    for (int i = 0; i < layerPanels.Length; i++)
                    {
                        for (int j = 0; j < featureMapImages[i].Length; j++)
                        {
                            if (featureMapImages[i][j].Image != null)
                            {
                                featureMapImages[i][j].Image.Dispose();
                            }
                            featureMapImages[i][j].Image = Helpers.ImageFromDoubleArray(convLayers[i].FeatureMaps[j].Output, 44, 44);
                        }
                    }

                    evt.Set();
                });

                if (this.InvokeRequired)
                {
                    this.BeginInvoke(uiThreadWork);
                }
                else
                {
                    uiThreadWork();
                }

                evt.WaitOne();
            }
        }

        private void prevButton_Click(object sender, EventArgs e)
        {
            testExample = testingExamples[textExampleIndex--];
            UpdateDisplay(testExample.Input, "");
        }

        private void nextButton_Click(object sender, EventArgs e)
        {
            testExample = testingExamples[textExampleIndex++];
            UpdateDisplay(testExample.Input, "");
        }
    }
}
