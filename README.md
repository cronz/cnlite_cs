# cnlite_cs
cnlite is a c# library to do chinese ocr based on project [chineseocr_lite](https://github.com/ouyanghuiyu/chineseocr_lite)

## Overview
* 1. cnlite reused onnx models pretrained by chineseocr_lite project
* 2. This project is translated from python code almost 99% percent from chineseocr_lite project

## Usage
* 1. download the project code
* 2. download onnx models file (crnn_lite_lstm.onnx and dbnet.onnx) from chineseocr_lite
* 3. use microsoft visual studio to download the necessary library
* 4. run the demo project
* 5. onnx models are located at application_root/conf/.

## Dependencies
* 1. .net framework 4.6.1
* 2. [onnxruntime](https://github.com/Microsoft/onnxruntime)
* 3. [clipper_library](https://sourceforge.net/projects/polyclipping/)
* 4. [EmguCV](http://www.emgu.com/wiki/index.php/Main_Page)

## How to use EmguCV
* 1. download EmguCV from [Download and installation](http://www.emgu.com/wiki/index.php/Download_And_Installation) page
* 2. installEmguCV
* 3. Copy library to your target path, the required vcrt dlls are included in the "x86" and "x64" folder. You will be ready as long as you copy all the unmanaged dlls in the "x86" and "x64" folder to the folder of executable
* 4. Be careful to configure target Platform based on "x84" or "x64" folder

<hr/>

## 说明
本项目是基于 chineseocr_lite 的源代码和onnx model，本项目只是翻译了原有的python代码为C#代码

## 使用方法
* 1. 下载本项目代码
* 2. 从chineseocr_lite 项目中下载onnx models
* 3. 在vs 2015/19中，用NuGet 下载onnxruntime
* 4. 注意，onnx训练文件请放到 conf 文件夹里面

## 使用环境
* 1. .net framework 4.6.1
* 2. [onnxruntime](https://github.com/Microsoft/onnxruntime)
* 3. [clipper_library](https://sourceforge.net/projects/polyclipping/)
* 4. [EmguCV](http://www.emgu.com/wiki/index.php/Main_Page)

## EmguCV 的使用
* 1. 本项目使用的EmguCV，是OpenCV 的C# API调用版本
* 2. 要先下载安装EmguCV
* 3. 注意要复制相应的 x64 或者 x86 文件夹的所有文件到编译路径下面。EmguCV的安装路径下面有对应的x86/x64文件夹，请将所有的动态库复制到C#工程编译路径(一般是bin\Debug)下面。请[参考](http://www.emgu.com/wiki/index.php/Download_And_Installation#Open_CV_unmanaged_dll)
