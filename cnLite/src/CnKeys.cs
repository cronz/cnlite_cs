using System;
using System.Collections.Generic;
using System.Linq;

namespace cnLite
{
    public class CnKeys
    {
        private static string Chinese = Resource.Keys;

        private static char[] AllChinese;

        public static char[] GetAllChinese()
        {
            if (AllChinese == null)
            {
                AllChinese = Chinese.ToArray();
            }

            return AllChinese;
        }
    }
}
