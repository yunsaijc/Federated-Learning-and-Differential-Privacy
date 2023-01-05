# README



## 项目说明

本项目为联邦学习+差分隐私的实现。支持同步、异步、半异步（默认）机制。

运行环境：python 3.6

本代码基于仓库https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning实现。

## 文件结构说明

项目文件结构及说明如下：

```
myFed
│  myFed.py			//项目main文件
│  plot.py			//用于作图的文件
│  predictParams.py	//用于估算参数(tau,t_avg,K等)的文件
│  README.md
│  requirements.txt	//依赖包与版本
│  
├─.idea
├─data				//存放数据集文件
├─FilesAndFigs		//需要的图
│  ├─p2     //文件夹名与PPT中的页码相对应，包含相应的数据文件与作图结果
│  ├─p3
│  ├─p4
│  ├─p5
│  ├─p6
│  └─p7
├─log       //代码运行记录
├─models    //网络模型相关文件
│  │  Fed.py        //全局更新函数
│  │  Nets.py       //网络模型定义函数
│  │  test.py       //测试准确率函数
│  │  Update.py     //本地更新函数
│  │  __init__.py
│
├─utils     //存放一些使用到的函数
│  │  dataset.py            //数据集有关函数
│  │  dp_mechanism.py       //与dp有关的函数
│  │  Functions.py          //其他函数
│  │  language_utils.py     //与特定语言模型有关的函数
│  │  options.py            //命令行参数设置函数
│  │  sampling.py           //数据集划分函数
│  │  __init__.py
│
├─oldVersion //仅仅实现了基本功能的旧版本，结构较混乱，可忽略
```

## 重要参数定义与说明

```
//参数定义
M		//每一轮参与全局训练的客户数量
N		//客户数量
K		//预计全局更新的轮数
tau		//陈旧度
epsilon	//隐私预算

//命令行重要参数
frac		//每一轮参与全局训练的客户所占的比例，即M/N
allParams	//布尔值，为真则自动让M和N遍历。N从10到1000进行遍历，步长为10；对于每个N，M从0.1到1.0进行遍历，步长为0.1。那么总共就是100*10=1000组M和N的值。
sec(0~9)	//allParams为真时可以设置。为了能够并行地遍历M和N，将N的值平均分为了10个sec，每个sec有100个part，每个part对应的是一组M和N的值。例如：sec=2, part=5, 对应的就是N=210, M=0.5；sec=2, part=16, 对应的就是N=220, M=0.6。
part(1~100)
selfLR		//布尔值，为真则使用自适应学习率
sync		//布尔值，为真则使用同步机制训练（默认使用半异步机制训练）
asy			//布尔值，为真则使用异步机制训练（默认使用半异步机制训练）
avg			//布尔值，为真则使用FedAvg的全局聚合方式（默认使用FedSA的全局聚合方式）

//各个机制的设定
myFed: frac = 0.2(iid) or 0.8(n-iid)
FedSA: frac = 0.2(iid) or 0.8(n-iid), dp_mechanism = no_dp
FedAsync: asy = true
FedAP: frac = 0.5
FedAF: avg = true
```

## 运行说明

运行命令举例：

```
//No DP
python FedSA.py --dataset mnist --iid --model cnn --dp_mechanism no_dp

//Gaussian Mechanism
python FedSA.py --dataset mnist --iid --model cnn --dp_mechanism Gaussian --dp_epsilon 10

//自己指定N和frac
python FedSA.py --dataset mnist --iid --model c nn --dp_mechanism Gaussian --dp_epsilon 10 --num_users 10 --frac 0.2

//N-iid
python FedSA.py --dataset mnist --model c nn --dp_mechanism Gaussian --dp_epsilon 10 --num_users 10 --frac 0.2

//自动遍历所有N和frac(从N=20, frac=1.0开始遍历)
python FedSA.py --dataset mnist --iid --model cnn --dp_mechanism Gaussian --dp_epsilon 10 --allParams true --sec 0 --part 20
```



