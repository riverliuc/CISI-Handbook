# Chinese Intelligent Speech Interaction Best Practice Collection

CISI stands for Chinese Intelligent Speech Interaction, it is a collection of best practices in conversaional AI area. Those best practices are implemented and accumulated by NVIDIA China Solution Architect Enterprise Team during their daily work when engaging with various customers on public cloud. This collection aims to teach cloud users how to leverage NVIDIA products to facilitate creation of Chinese conversational AI applications, improve the efficiency of related workloads, and help them easily using GPU resources on cloud. CISI contains end-to-end opensourced model training and inference serving recipes for ASR, NLU, as well as TTS, which are based on various NVIDIA exisiting solutions including [NeMo](https://github.com/NVIDIA/NeMo), [TensorRT](https://developer.nvidia.com/tensorrt), [Triton](https://github.com/triton-inference-server/server), [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples), and so on. 

Intelligent speech interaction is widely used in different scenarios, such as voice assistant, smart speakers, intelligent customer service and so on. Some of them will use either single ASR or single TTS, some of them will use all three components to support full-intelligent conversation with human. No matter which case you are woking on, we provide a quick start point to help you obtain the AI power you need at the beginning. For example, if you want to train an Chinese ASR/NLU/TTS model with your own dataset, CISI offers you training recipes (based on NeMo/PyTorch/TensorFlow), including data preparation guidelines and training scripts; if you want to deploy Chinese ASR/NLU/TTS service on cloud with trained modelss, we also offer you a inference serving approaches based on [TensorRT](https://developer.nvidia.com/tensorrt) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) that allows you easily deploy models accelerated by NVIDIA which have remarkable performance. With CISI, users no longer need to work from scratch to get a model or deploy it as service, and all the best practices are open-sourced, users can apply any modifications to them to achieve specific targets.

As mentioned, CISI is a quick start point for users. When users get familiar with NVIDIA solutions and want some advanced features in intelligent speech interaction, they can switch to use [NeMo](https://github.com/NVIDIA/NeMo) for model training and [RIVA](https://developer.nvidia.com/riva) for building complex conversational AI applications by themselves.

Here we summarize the advantages of CISI:
- Fully open-sourced, can be customized as white box;
- Convenient training/finetuning recipes, users can obtain models fit to specific business cases;
- Easy-to-use serving best practice, able to quickly deploy trained models as inference service;
- Remarkable performance, served models are optimized by NVIDIA acceleration technologies;
- Aiming at Chinese language.

## Key Features

NVIDIA CISI consists of three major parts that contribute to building a conversational AI application: ASR, NLU and TTS. For each part, we provide both training best practice (for conviniently training/finetuning models) and inference best practice (for quickly serving trained models). The following are the key features of each part:

- ASR
  - Multi-GPU & multi-node distributed training recipes on public cloud
  - INT8 PTQ & QAT quantization recipe for non-streaming ASR based on QuartzNet
  - E2E streaming and non-stremaing ASR serving best practice based on WeNet
  - Pretrained Chinese non-streaming model and streaming model for users to finetune
  - Multi-node non-streaming ASR serving recipe based on K8S + Triton on Ali Cloud and Tencent Cloud
  - ASR service performance benchmark on public cloud
  - A web demo for streaming ASR service
- NLU
  - Multi-GPU BERT training recipe
  - Dialog manger (DM) training recipe based on RASA
  - Simple datasets for training NLU model and DM
  - TRT-accelerated BERT for NLU inference
  - Single-node NLU inference deployment recipe based on RASA and Triton
  - Reference python client for accessing NLU service
  - QAT & PTQ recipe for BERT based on TensorRT
- TTS
  - Multi-GPU training recipe on public cloud
  - Dataset preparation guideline
  - A development guide for a Chinese prosody model
  - Non-streaming TTS serving recipe for deploying on public cloud
  - Support multiple vocoders (WaveGlow & WaveRNN)
  - Streaming response (return TTS result back to client in streaming way, chunk by chunk)

## Repository Structure

The structure of this repository is organized as below:

- [asr](asr/README.md): Training and inference best practice for both gitstreaming and non-streaming ASR service
  - `quartznet_offline`: end2end non-streaming ASR best practice based on QuartzNet
  - `kaldi_online`: end2end streaming ASR best practice based on Kaldi
- [nlu](nlu/README.md): Training and inference best practice for task-oriented bot NLU service
- [tts](tts/README.md): Training and inference best practice for non-streaming TTS service
  - `train`: best practice for training prosodic prediction + Tacotron2 + WaveGlow TTS models
  - `inference`: serving best practice for deploying trained TTS models based on Triton
