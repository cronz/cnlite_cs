# cnlite_cs
cnlite is a c# library to do chinese ocr based on project [chineseocr_lit](https://github.com/ouyanghuiyu/chineseocr_lite)

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


<hr/>

## 说明
本项目是基于 chineseocr_lite 的源代码和onnx model，本项目只是翻译了原有的python代码为C#代码

## 使用方法
*1. 下载本项目代码
*2. 从chineseocr_lite 项目中下载onnx models
*3. 在vs 2015/19中，用NuGet 下载onnxruntime
*4. 注意，onnx训练文件请放到 conf 文件夹里面

## 使用环境
*1. .net framework 4.6.1
*2. onnxruntime
*3. clipper_library
