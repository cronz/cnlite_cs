using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace cnLite
{
    public class CRNNHandle : IDisposable
    {
        private Dictionary<int, char> chineseKeys;

        public CRNNHandle()
        {
            chineseKeys = new Dictionary<int, char>();
            PrepareChinese();
        }

        ~CRNNHandle() { }

        public void Dispose()
        {

        }

        private void PrepareChinese()
        {
            int idx = 1;
            char[] allcn = CnKeys.GetAllChinese();
            foreach (char cn in allcn)
            {
                chineseKeys.Add(idx, cn);
                idx++;
            }
        }

        public string Run(InferenceSession session, Tensor<float> oneTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", oneTensor)
            };

            string res = string.Empty;

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
            {
                // Postprocess to get predictions
                var resultsArray = results.ToArray();
                res = Decode(resultsArray);
            }

            return res;
        }

        private string Decode(DisposableNamedOnnxValue[] results)
        {
            Tensor<float> values = results[0].Value as Tensor<float>;
            if (values == null)
            {
                return "";
            }
            int allValues = values.Count();
            if (values.Dimensions.IsEmpty || allValues == 0)
            {
                return "";
            }

            int[] dims = values.Dimensions.ToArray();

            if (dims[0] == 0)
            {
                return "";
            }

            float max = -100;
            int maxOfArr = 0;

            List<int> idxMax = new List<int>();
            int rowCnt = allValues / dims[0];

            for (int idx = 0; idx < allValues; idx += rowCnt)
            {
                max = -100;
                maxOfArr = 0;

                for (int idy = idx; idy < rowCnt + idx; idy++)
                {
                    if (max < values.GetValue(idy))
                    {
                        max = values.GetValue(idy);
                        maxOfArr = idy - idx;
                    }
                }

                if (maxOfArr > 0)
                {
                    idxMax.Add(maxOfArr);
                }
            }

            string ret = string.Empty;
            foreach (int val in idxMax)
            {
                if (chineseKeys.ContainsKey(val + 1))
                {
                    ret = string.Format("{0}{1}", ret, chineseKeys[val + 1]);
                }
            }

            return ret;
        }
    }
}
