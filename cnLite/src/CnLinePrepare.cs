using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;

using Microsoft.ML.OnnxRuntime.Tensors;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace cnLite
{
    class CnLinePrepare
    {
        private Dictionary<int, List<Point>> box;
        private Mat rgbMat;

        public CnLinePrepare(Mat rgbMat, Dictionary<int, List<Point>> box)
        {
            this.rgbMat = rgbMat;
            this.box = box;
        }

        ~CnLinePrepare() { }

        public List<Tensor<float>> GetTensors()
        {
            if (box == null)
            {
                return null;
            }

            if (box.Count == 0)
            {
                return null;
            }

            List<Tensor<float>> tensors = new List<Tensor<float>>();

            foreach (List<Point> theRealBox in box.Values)
            {
                Mat roiMat = new Mat();
                GetRoiFromBox(roiMat, theRealBox);
                Tensor<float> tensorForBox = GetTensorInputFromImg(roiMat);
                tensors.Add(tensorForBox);
            }

            return tensors;
        }

        private void GetRoiFromBox(Mat roiMat, List<Point> theRealBox)
        {
            if (theRealBox == null || theRealBox.Count != 4)
            {
                return;
            }

            int left = 2400, right = 0, top = 4000, bottom = 0;

            foreach (Point pt in theRealBox)
            {
                if (pt.X < left)
                {
                    left = pt.X;
                }

                if (pt.Y < top)
                {
                    top = pt.Y;
                }

                if (pt.X > right)
                {
                    right = pt.X;
                }

                if (pt.Y > bottom)
                {
                    bottom = pt.Y;
                }
            }

            Image<Rgb, byte> img = rgbMat.ToImage<Rgb, byte>();
            img.ROI = new Rectangle(left, top, right - left + 1, bottom - top + 1);

            List<PointF> dstBox = new List<PointF>();

            foreach (Point pt in theRealBox)
            {
                Point npt = new Point(pt.X - left, pt.Y - top);
                dstBox.Add(npt);
            }

            int imgCropWidth = LenthOfPoints(theRealBox[0], theRealBox[1]);
            int imgCropHeight = LenthOfPoints(theRealBox[0], theRealBox[3]);

            PointF[] trans_corner = new PointF[4];
            trans_corner[0] = new PointF(0, 0);
            trans_corner[1] = new PointF(imgCropWidth, 0);
            trans_corner[2] = new PointF(imgCropWidth, imgCropHeight);
            trans_corner[3] = new PointF(0, imgCropHeight);
            Mat persp = CvInvoke.GetPerspectiveTransform(dstBox.ToArray<PointF>(), trans_corner);
            CvInvoke.WarpPerspective(img, roiMat, persp, new Size(imgCropWidth, imgCropHeight),
                Inter.Nearest, Warp.Default, BorderType.Constant);
        }

        private int LenthOfPoints(Point pt1, Point pt2)
        {
            int dx = pt2.X - pt1.X;
            int dy = pt2.Y - pt1.Y;

            return (int)Math.Sqrt(dx * dx + dy * dy);
        }

        private Tensor<float> GetTensorInputFromImg(Mat srcMat)
        {
            float DIVIDE_VAL = 127.5f;

            double scale = srcMat.Height * 1.0 / 32;
            int dstWidth = (int)(srcMat.Width / scale);

            Mat imgResized = new Mat(32, dstWidth, DepthType.Cv32F, 3);
            CvInvoke.Resize(srcMat, imgResized, new Size(dstWidth, 32));

            Tensor<float> inputs = new DenseTensor<float>(new[] { 1, 3, imgResized.Height, imgResized.Width });

            Image<Rgb, byte> outImg = imgResized.ToImage<Rgb, byte>();
            int rows = outImg.Rows;
            int cols = outImg.Cols;
            byte[,,] data = outImg.Data;

            try
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        float rr = (float)((data[i, j, 0] - DIVIDE_VAL) / DIVIDE_VAL);
                        float gg = (float)((data[i, j, 1] - DIVIDE_VAL) / DIVIDE_VAL);
                        float bb = (float)((data[i, j, 2] - DIVIDE_VAL) / DIVIDE_VAL);

                        inputs[0, 0, i, j] = rr;
                        inputs[0, 1, i, j] = gg;
                        inputs[0, 2, i, j] = bb;
                    }
                }
            }
            catch (Exception ex)
            {
                 Console.Write("GetTensorInputFromImg, " + ex.Message);
            }

            outImg.Dispose();

            return inputs;
        }
    }
}
