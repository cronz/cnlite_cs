using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;

using Microsoft.ML.OnnxRuntime;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using ClipperLib;

namespace cnLite
{
    public class LineBoxDecode
    {
        private float minSize = 3;
        private float thresh = 0.3f;
        private float boxThresh = 0.5f;
        private int maxCandidates = 1000;
        private float unclipRatio = 2.0f;

        private int Width;
        private int Height;

        private Mat segmentFloatMat;

        public Dictionary<int, List<Point>> Boxes { get; set; }

        public LineBoxDecode(int width, int height)
        {
            Width = width;
            Height = height;
            segmentFloatMat = new Mat();
            Boxes = new Dictionary<int, List<Point>>();
        }

        ~LineBoxDecode() { }

        public void Do(DisposableNamedOnnxValue[] outs, int width, int height)
        {
            Mat segment = GetSegmentation(outs, width, height);
            BoxesFromImage(segment, width, height);
        }

        private void BoxesFromImage(Mat segment, int width, int height)
        {
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(segment, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                int numContours = Math.Min(contours.Size, maxCandidates);

                for (int idx = 0; idx < contours.Size; idx++)
                {
                    float minS = 0;
                    List<PointF> box = GetMiniBox(contours[idx], out minS);

                    if (minS < minSize)
                    {
                        continue;
                    }

                    double score = GetScore(contours[idx]);
                    if (score < boxThresh)
                    {
                        continue;
                    }

                    List<Point> solutions = Unclip(box, unclipRatio);
                    if (solutions == null)
                    {
                        continue;
                    }

                    List<PointF> secondBox = GetMiniBox(solutions, out minS);
                    if (minS < minSize + 2)
                    {
                        continue;
                    }

                    List<Point> finalPoints = new List<Point>();

                    Boxes.Add(idx, finalPoints);
                    foreach (PointF pt in secondBox)
                    {
                        int ptx = (int)(pt.X / width * Width);
                        int pty = (int)(pt.Y / height * Height);
                        if (ptx + 1 > Width)
                        {
                            ptx = Width - 1;
                        }

                        if (pty + 1 > Height)
                        {
                            pty = Height - 1;
                        }

                        Point alpt = new Point(ptx, pty);
                        finalPoints.Add(alpt);
                    }
                }
            }
        }

        private double LengthOfPoints(List<PointF> box)
        {
            double length = 0;

            PointF pt = box[0];
            double x0 = pt.X;
            double y0 = pt.Y;
            double x1 = 0, y1 = 0, dx = 0, dy = 0;
            box.Add(pt);

            int count = box.Count;
            for (int idx = 1; idx < count; idx++)
            {
                PointF pts = box[idx];
                x1 = pts.X;
                y1 = pts.Y;
                dx = x1 - x0;
                dy = y1 - y0;

                length += Math.Sqrt(dx * dx + dy * dy);

                x0 = x1;
                y0 = y1;
            }

            box.RemoveAt(count - 1);
            return length;
        }

        private List<Point> Unclip(List<PointF> box, float unclip_ratio)
        {
            List<IntPoint> theCliperPts = new List<IntPoint>();
            foreach (PointF pt in box)
            {
                IntPoint a1 = new IntPoint((int)pt.X, (int)pt.Y);
                theCliperPts.Add(a1);
            }

            float area = Math.Abs(SignedPolygonArea(box.ToArray<PointF>()));
            double length = LengthOfPoints(box);
            double distance = area * unclip_ratio / length;

            ClipperOffset co = new ClipperOffset();
            co.AddPath(theCliperPts, JoinType.jtRound, EndType.etClosedPolygon);
            List<List<IntPoint>> solution = new List<List<IntPoint>>();
            co.Execute(ref solution, distance);
            if (solution.Count == 0)
            {
                return null;
            }

            List<Point> retPts = new List<Point>();
            foreach (IntPoint ip in solution[0])
            {
                retPts.Add(new Point((int)ip.X, (int)ip.Y));
            }

            return retPts;
        }

        private float SignedPolygonArea(PointF[] Points)
        {
            // Add the first point to the end.
            int num_points = Points.Length;
            PointF[] pts = new PointF[num_points + 1];
            Points.CopyTo(pts, 0);
            pts[num_points] = Points[0];

            // Get the areas.
            float area = 0;
            for (int i = 0; i < num_points; i++)
            {
                area +=
                    (pts[i + 1].X - pts[i].X) *
                    (pts[i + 1].Y + pts[i].Y) / 2;
            }

            return area;
        }

        private double GetScore(VectorOfPoint contours)
        {
            short xmin = 9999;
            short xmax = 0;
            short ymin = 9999;
            short ymax = 0;

            try
            {
                foreach (Point point in contours.ToArray())
                {
                    if (point.X < xmin)
                    {
                        //var xx = nd[point.X];
                        xmin = (short)point.X;
                    }

                    if (point.X > xmax)
                    {
                        xmax = (short)point.X;
                    }

                    if (point.Y < ymin)
                    {
                        ymin = (short)point.Y;
                    }

                    if (point.Y > ymax)
                    {
                        ymax = (short)point.Y;
                    }
                }

                int roiWidth = xmax - xmin + 1;
                int roiHeight = ymax - ymin + 1;

                Image<Gray, float> bitmap = segmentFloatMat.ToImage<Gray, float>();
                Image<Gray, float> roiBitmap = new Image<Gray, float>(roiWidth, roiHeight);
                float[,,] dataFloat = bitmap.Data;
                float[,,] data = roiBitmap.Data;

                for (int j = ymin; j < ymin + roiHeight; j++)
                {
                    for (int i = xmin; i < xmin + roiWidth; i++)
                    {
                        try
                        {
                            data[j - ymin, i - xmin, 0] = dataFloat[j, i, 0];
                        }
                        catch (Exception ex2)
                        {
                            Console.WriteLine(ex2.Message);
                        }
                    }
                }

                Mat mask = Mat.Zeros(roiHeight, roiWidth, DepthType.Cv8U, 1);
                List<Point> pts = new List<Point>();
                foreach (Point point in contours.ToArray())
                {
                    pts.Add(new Point(point.X - xmin, point.Y - ymin));
                }

                using (VectorOfPoint vp = new VectorOfPoint(pts.ToArray<Point>()))
                using (VectorOfVectorOfPoint vvp = new VectorOfVectorOfPoint(vp))
                {
                    CvInvoke.FillPoly(mask, vvp, new MCvScalar(1));
                }

                return CvInvoke.Mean(roiBitmap, mask).V0;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }

            return 0;
        }

        private List<PointF> GetMiniBox(List<Point> contours, out float minS)
        {
            VectorOfPoint vop = new VectorOfPoint();
            vop.Push(contours.ToArray<Point>());
            return GetMiniBox(vop, out minS);
        }

        private List<PointF> GetMiniBox(VectorOfPoint contours, out float minS)
        {
            List<PointF> box = new List<PointF>();
            RotatedRect rrect = CvInvoke.MinAreaRect(contours);
            PointF[] points = CvInvoke.BoxPoints(rrect);
            minS = Math.Min(rrect.Size.Width, rrect.Size.Height);

            List<PointF> thePoints = new List<PointF>(points);
            thePoints.Sort(CompareByX);

            int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
            if (thePoints[1].Y > thePoints[0].Y)
            {
                index_1 = 0;
                index_4 = 1;
            }
            else
            {
                index_1 = 1;
                index_4 = 0;
            }

            if (thePoints[3].Y > thePoints[2].Y)
            {
                index_2 = 2;
                index_3 = 3;
            }
            else
            {
                index_2 = 3;
                index_3 = 2;
            }

            box.Add(thePoints[index_1]);
            box.Add(thePoints[index_2]);
            box.Add(thePoints[index_3]);
            box.Add(thePoints[index_4]);

            return box;
        }

        public static int CompareByX(PointF left, PointF right)
        {
            if (left == null && right == null)
            {
                return 1;
            }

            if (left == null)
            {
                return 0;
            }

            if (right == null)
            {
                return 1;
            }

            if (left.X > right.X)
            {
                return 1;
            }

            if (left.X == right.X)
            {
                return 0;
            }

            return -1;
        }

        private Mat GetSegmentation(DisposableNamedOnnxValue[] outs, int width, int height)
        {
            Image<Gray, byte> outImg = new Image<Gray, byte>(width, height);
            Image<Gray, float> bitmap = new Image<Gray, float>(width, height);

            byte[,,] data = outImg.Data;
            float[,,] dataFloat = bitmap.Data;

            try
            {
                float[] boxes = outs[0].AsEnumerable<float>().ToArray();
                List<byte> imgData = new List<byte>();

                foreach (float x in boxes)
                {
                    byte val = 0;

                    if (x > thresh)
                    {
                        val = 255;
                    }

                    imgData.Add(val);
                }

                int idx = 0;

                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        data[i, j, 0] = imgData[idx];
                        dataFloat[i, j, 0] = boxes[idx++];
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Write("GetSegmentation, " + ex.Message);
            }

            bitmap.Mat.CopyTo(segmentFloatMat);
            return outImg.Mat;
        }
    }
}
