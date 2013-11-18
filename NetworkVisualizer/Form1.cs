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
using System.Drawing.Imaging;

namespace NetworkVisualizer
{
    public partial class Form1 : Form
    {
        private NetworkManager m_Manager;
        private BackgroundWorker m_Worker;

        private FlowLayoutPanel[] layerPanels;
        private PictureBox[][] featureMapImages;

        private ITrainingSetProvider provider;
        private TrainingExample testExample;

        private int textExampleIndex = 0;

        private ManualResetEvent evt = new ManualResetEvent(false);

        public Form1()
        {
            InitializeComponent();
            label2.Visible = false;
            progressBar1.Visible = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            button1.Enabled = false;

            ITrainingSetProvider trainingSetProvider = new MNISTTrainingSetProvider();
            provider = trainingSetProvider;
            testExample = trainingSetProvider.GetTestingExamples()[textExampleIndex];
            pictureBox1.Image = ImageFromDoubleArray(testExample.Input, 48, 48);
            pictureBox1.BeginInvoke(new Action(() =>
            {
                pictureBox1.Image = ImageFromDoubleArray(testExample.Input, 28, 28);
            }));

            //new ConvolutionalLayer(ActivationFunctionType.Tanh, 4, 20, 2),
            //                                   new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 60, 2),
            //                                   new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 120, 1),
            //120000 iters, Tanh, 1 layer of 300, 0.005 = ~13% error
            //120000 iters, Sigmoid, 1 layer of 300, 0.005 = ~13% error
            m_Manager = new NetworkManager("bestsofar",
                                           trainingSetProvider,
                                           new InputLayer(28 * 28),
                                           new List<HiddenLayer>()
                                           {
                                               new ConvolutionalLayer(ActivationFunctionType.Tanh, 2, 20, 2),
                                               new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 60, 2),
                                               new ConvolutionalLayer(ActivationFunctionType.Tanh, 5, 120, 1),
                                               new HiddenLayer(ActivationFunctionType.Tanh, 800),
                                           },
                                           new OutputLayer(ActivationFunctionType.Softmax, 10),
                                           0.00025); //0.005

            try
            {
                SetupDisplay(m_Manager.GetNetwork(), trainingSetProvider);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            label2.Visible = true;
            progressBar1.Visible = true;
            
            TrainNetwork(0.001);
        }

        private void TrainNetwork(double errorThreshold)
        {
            m_Worker = new BackgroundWorker();
            m_Worker.WorkerSupportsCancellation = true;
            m_Worker.WorkerReportsProgress = true;

            int trainingIters = 120000;

            double avgError = 5;
            int epoch = 0;

            m_Worker.DoWork += new DoWorkEventHandler((o, args) =>
            {
                m_Manager.TrainNetwork(trainingIters, false, (network, error) =>
                {
                    
                    if (m_Worker.CancellationPending)
                    {
                        m_Worker.Dispose();
                        return false;
                    }

                    avgError *= 24;
                    avgError += error;
                    avgError /= 25;

                    if (double.IsNaN(avgError) || double.IsNaN(error))
                    {
                        MessageBox.Show("NAN. error: "+error);
                    }

                    epoch++;

                    if (epoch % 50 == 0 || avgError <= errorThreshold)
                    {
                        lblAvgError.BeginInvoke(new Action(() =>
                        {
                            lblAvgError.Text = "" + avgError;
                        }));

                        UpdateDisplay(network, testExample.Input);
                    }

                    if (epoch % 10 == 0)
                    {
                        double temp = ((double)epoch / (double)trainingIters) * 100;

                        m_Worker.ReportProgress((int)temp);
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
                TestNetwork();
            });

            m_Worker.RunWorkerAsync();
        }

        private void TestNetwork()
        {
            int totalTests = 0;
            int totalCorrect = 0;

            m_Worker = new BackgroundWorker();
            m_Worker.WorkerSupportsCancellation = true;
            m_Worker.WorkerReportsProgress = true;

            int numIters = provider.GetTestingExamples().Count();
            int epoch = 0;

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

                        int num = Helpers.GetMaxValueIndex(result);
                       
                        int expectedNum = Helpers.GetMaxValueIndex(expected);
                        
                        totalTests++;

                        if (expectedNum == num)
                        {
                            totalCorrect++;
                            //System.Threading.Thread.Sleep(20);
                        }
                        else
                        {
                            //Slow down to show incorrect result
                            //System.Threading.Thread.Sleep(500);
                        }

                        epoch++;

                        if (epoch % 10 == 0)
                        {
                            double temp = ((double)epoch / (double)numIters) * 100;
                            m_Worker.ReportProgress((int)temp);
                        }

                        if (epoch % 100 == 0)
                        {
                            label1.BeginInvoke(new Action(() =>
                            {
                                label1.Text = "" + num;
                            }));

                            pictureBox1.BeginInvoke(new Action(() =>
                            {
                                if (pictureBox1.Image != null)
                                {
                                    pictureBox1.Image.Dispose();
                                }
                                pictureBox1.Image = ImageFromDoubleArray(input, 28, 28);
                            }));

                            UpdateDisplay(m_Manager.GetNetwork(), input);
                        }

                        //System.Threading.Thread.Sleep(500);

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
                button1.Enabled = true;
                MessageBox.Show(totalCorrect + " out of " + totalTests + " correct");
            });

            label2.Visible = true;
            label2.Text = "Testing...";
            progressBar1.Visible = true;
            progressBar1.Value = 0;

            m_Worker.RunWorkerAsync();
        }

        private void SetupDisplay(NeuralNetwork.NeuralNetwork network, ITrainingSetProvider provider)
        {
            List<ConvolutionalLayer> convLayers = network.HiddenLayers.Where(l => l is ConvolutionalLayer).Select(l => (ConvolutionalLayer)l).ToList();

            layerPanels = new FlowLayoutPanel[convLayers.Count];
            featureMapImages = new PictureBox[convLayers.Count][];

            for (int i = 0; i < layerPanels.Length; i++)
            {
                featureMapImages[i] = new PictureBox[convLayers[i].NumFeatureMaps];

                layerPanels[i] = new FlowLayoutPanel();
                layerPanels[i].FlowDirection = FlowDirection.TopDown;
                layerPanels[i].BorderStyle = BorderStyle.Fixed3D;
                layerPanels[i].Width = 150;
                layerPanels[i].Height = 300;
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
        }

        private void UpdateDisplay(NeuralNetwork.NeuralNetwork network, double[] input)
        {
            List<ConvolutionalLayer> convLayers = network.HiddenLayers.Where(l => l is ConvolutionalLayer).Select(l => (ConvolutionalLayer)l).ToList();

            network.GetResult(input);

            evt.Reset();
            this.BeginInvoke(new Action(() =>
            {
                for (int i = 0; i < layerPanels.Length; i++)
                {
                    for (int j = 0; j < featureMapImages[i].Length; j++)
                    {
                        if (featureMapImages[i][j].Image != null)
                        {
                            featureMapImages[i][j].Image.Dispose();
                        }
                        featureMapImages[i][j].Image = ImageFromDoubleArray(convLayers[i].FeatureMaps[j].Output, 44, 44);
                    }
                }

                evt.Set();
            }));
            evt.WaitOne();
        }

        private Image ImageFromDoubleArray(double[] data, int desiredWidth, int desiredHeight)
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
            

            //PixelFormat pixelFormat = PixelFormat.Format8bppIndexed;
            //int stride = origWidth;

            //Bitmap bmp = new Bitmap(origWidth, origWidth, pixelFormat);
            //System.Drawing.Imaging.BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, origWidth, origWidth), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);

            //IntPtr ptr = bmpData.Scan0;

            //Marshal.Copy(byteData, 0, ptr, origWidth * origWidth);

            //bmp.UnlockBits(bmpData);

            //Bitmap resized = new Bitmap(bmp, new Size(width, height));

            //return bmp;
            return ResizeBitmap(bmp, origWidth * 3, origWidth * 3);
        }

        private Bitmap ResizeBitmap(Bitmap sourceBMP, int width, int height)
        {
            Bitmap result = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format16bppRgb555);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.DrawImage(sourceBMP, 0, 0, width, height);
            }
            sourceBMP.Dispose();
            return result;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            m_Worker.CancelAsync();
        }

        private void prevButton_Click(object sender, EventArgs e)
        {
            testExample = provider.GetTestingExamples()[textExampleIndex--];
            pictureBox1.BeginInvoke(new Action(() =>
            {
                pictureBox1.Image = ImageFromDoubleArray(testExample.Input, 28, 28);
            }));
        }

        private void nextButton_Click(object sender, EventArgs e)
        {
            testExample = provider.GetTestingExamples()[textExampleIndex++];
            pictureBox1.BeginInvoke(new Action(() =>
            {
                pictureBox1.Image = ImageFromDoubleArray(testExample.Input, 28, 28);
            }));
        }
    }
}
