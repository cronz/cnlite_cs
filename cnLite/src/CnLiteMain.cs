using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace cnLite
{
    public class CnLiteMain
    {
        private string modelFilePath;
        private string crNNModelFilePath;

        public CnLiteMain(string modelFilePath, string crNNModelFilePath)
        {
            this.modelFilePath = modelFilePath;
            this.crNNModelFilePath = crNNModelFilePath;
        }

        private Tensor<float> GetTensorInputFromImg(Mat srcMat)
        {
            Tensor<float> inputs = new DenseTensor<float>(new[] { 1, 3, srcMat.Height, srcMat.Width });

            Image<Rgb, byte> outImg = srcMat.ToImage<Rgb, byte>();
            int rows = outImg.Rows;
            int cols = outImg.Cols;
            byte[,,] data = outImg.Data;

            try
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        float rr = (float)((data[i, j, 0] * 1.0 / 255.0 - 0.485) / 0.229);
                        float gg = (float)((data[i, j, 1] * 1.0 / 255.0 - 0.456) / 0.224);
                        float bb = (float)((data[i, j, 2] * 1.0 / 255.0 - 0.406) / 0.225);

                        inputs[0, 0, i, j] = rr;
                        inputs[0, 1, i, j] = gg;
                        inputs[0, 2, i, j] = bb;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("LeftRightColorImg::Bind, " + ex.Message);
            }

            outImg.Dispose();

            return inputs;
        }

        private Tensor<float> GetInputs(string imageFilePath, Mat rgbMat, out int new_h, out int new_w)
        {
            int short_size = 960;
            // Read image
            Mat imageMat = CvInvoke.Imread(imageFilePath, ImreadModes.Color);
            CvInvoke.CvtColor(imageMat, rgbMat, ColorConversion.Bgr2Rgb);
            imageMat.Dispose();

            double scale_h = 0, tar_w = 0, scale_w = 0, tar_h = 0;

            if (rgbMat.Height < rgbMat.Width)
            {
                scale_h = short_size * 1.0 / rgbMat.Height;
                tar_w = rgbMat.Width * scale_h * 1.0;
                tar_w = tar_w - tar_w % 32;
                tar_w = Math.Max(32, tar_w);
                scale_w = tar_w / rgbMat.Width;
            }
            else
            {
                scale_w = short_size * 1.0 / rgbMat.Width;
                tar_h = rgbMat.Height * scale_w * 1.0;
                tar_h = tar_h - tar_h % 32;
                tar_h = Math.Max(32, tar_h);
                scale_h = tar_h / rgbMat.Height;
            }

            new_h = (int)(scale_h * rgbMat.Height);
            new_w = (int)(scale_w * rgbMat.Width);

            Mat imgResized = new Mat(new_h, new_w, DepthType.Cv32F, 3);
            CvInvoke.Resize(rgbMat, imgResized, new Size(new_w, new_h));

            return GetTensorInputFromImg(imgResized);
        }

        public List<string> DoOcr(string imageFilePath)
        {
            Mat rgbMat = new Mat();

            int new_h, new_w;
            Tensor<float> input = GetInputs(imageFilePath, rgbMat, out new_h, out new_w);

            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input0", input)
            };

            try
            {
                LineBoxDecode decode = new LineBoxDecode(rgbMat.Width, rgbMat.Height);
                // Run inference modelFilePath
                using (var session = new InferenceSession(modelFilePath))
                {
                    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
                    {
                        var resultsArray = results.ToArray();
                        decode.Do(resultsArray, new_w, new_h);
                    }
                }

                CnLinePrepare clp = new CnLinePrepare(rgbMat, decode.Boxes);
                List<Tensor<float>> tensors = clp.GetTensors();

                List<string> fresults = new List<string>();
                using (var crnn = new CRNNHandle())
                {
                    //crNNModelFilePath
                    using (var session = new InferenceSession(crNNModelFilePath))
                    {
                        foreach (Tensor<float> oneTensor in tensors)
                        {
                            fresults.Add(crnn.Run(session, oneTensor));
                        }
                    }
                }

                return fresults;
                /*
                 *  the following code to draw box for each line of chinese 
                foreach (List<Point> boxes in box.Values)
                {
                    DrawBox(rgbMat, boxes);
                }

                CvInvoke.Imwrite(outImageFilePath, rgbMat);
                */
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }

            return null;
        }

        private void DrawBox(Mat src, List<Point> box)
        {
            if (box == null || box.Count == 0)
            {
                return;
            }

            Point pt0 = box[0];
            box.Add(pt0);
            int cnt = box.Count;
            int x = pt0.X;
            int y = pt0.Y;

            for (int idx = 1; idx < box.Count; idx++)
            {
                Point tmp = new Point(x, y);
                Point pt1 = box[idx];
                CvInvoke.Line(src, pt1, tmp, new MCvScalar(0, 0, 255));
                x = pt1.X;
                y = pt1.Y;
            }

            box.RemoveAt(cnt - 1);
        }
    }
}
