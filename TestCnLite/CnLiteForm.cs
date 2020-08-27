using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;

using cnLite;

namespace TestCnLite
{
    public partial class CnLiteForm : Form
    {
        private CnLiteMain clm;

        public CnLiteForm()
        {
            InitializeComponent();
        }

        private void CnLiteForm_Load(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory;
            string currPath = Directory.GetParent(path).FullName;
            string modelFilePath = path + "conf\\dbnet.onnx";
            string crNNModelFilePath = path + "conf\\crnn_lite_lstm.onnx";
            
            clm = new CnLiteMain(modelFilePath, crNNModelFilePath);
        }

        private void RunTest()
        {
            listBox1.Items.Clear();
            using (var dlg = new OpenFileDialog())
            {
                dlg.Multiselect = false;
                dlg.Filter = "jpeg files (*.jpg)|*.jpg";
                if (dlg.ShowDialog() == DialogResult.OK && !string.IsNullOrEmpty(dlg.FileName))
                {
                    List<string> results = clm.DoOcr(dlg.FileName);
                    if (results == null)
                    {
                        return;
                    }

                    bool first = true;
                    foreach (string res in results)
                    {
                        if (first)
                        {
                            listBox1.Items.Add(res);
                            first = false;
                        }
                        else
                        {
                            listBox1.Items.Insert(0, res);
                        }

                    }
                }
            }
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            RunTest();
        }
    }
}
