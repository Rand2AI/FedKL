# FedKL: Gradient Leakage Defense with Key-Lock Module for Federated Learning

## Introduction

Federated learning (FL) is a promising modern technology for privacy-preserving machine learning, where private data is kept locally to perform secure local computations. In contrast, the gradient of the local model is exchanged instead between the local client and the third-party parameter server. Recently, gradient leakage attack in the FL system has been absorbing great attention. Many studies have revealed that privacy is compromised and that sensitive information can be stolen by malicious from the public shared gradient, i.e. DLG, GRNN, and GGL. Apart from most of the current methods, in this paper, we explore a new perspective of method on defending Federated Learning Gradient Leakage by securing arbitrary model architectures with a private key-lock module (FedKL). Only the locked gradient is transferred to the parameter server for aggregating the global model. The proposed FedKL is robust against gradient leakage attacks and the gist of the key-lock module is to design and train the model in a way that without the private information of the key-lock module: a) it is infeasible to reconstruct private training data from the shared gradient; b) the inference performance of the global model will be significantly undermined. Theoretical analysis of why the gradient can leak private information is discussed in this paper, and how the proposed method defends the attack is proved as well. Empirical evaluations on many popular benchmarks are conducted, demonstrating that our method can not only have a minor impact on the model inference performance but also prevent the gradient leakage effectively.

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/illustration.png" width="600px"/></div>

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/key-lock.png" width="600px"/></div>

## Requirements

python==3.6.9

torch==1.4.0

torchvision==0.5.0

numpy==1.18.2

tqdm==4.45.0

...

## Examples

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/example_1.png" width="600px"/></div>

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/example_2.png" width="400px"/></div>

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/example_3.png" width="400px"/></div>

## Performance

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/accuracy.png" width="1000px"/></div>

<div align=center><img src="https://github.com/Rand2AI/FedKL/blob/main/images/comparison.png" width="1000px"/></div>

## Citation

If you find this work helpful for your research, please cite the following paper:

    @article{,
             author = {Ren, Hanchi and Deng, Jingjing and Xie, Xianghua and Ma, Xiaoke and Ma, Jianfeng},
             title = {FedKL: Gradient Leakage Defense with Key-Lock Module for Federated Learning},
             year = {2023},
             publisher = {},
             address = {},
             issn = {},
             url = {},
             journal = {}
            }
