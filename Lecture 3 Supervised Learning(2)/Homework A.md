# Homework

![image-20250801150806874](./assets/image-20250801150806874.png)

>F:应该是避免了梯度消失
>
>F:可以使用但是尽量不一起使用(Dropout会改变BN依赖的均值和方差)
>
>>   即可以先BN->然后Dropout
>>
>>   但是先进的网络架构已经默认不使用Dropout,BN的正则化很强了
>
>F:BN并不是GN的特例
>
>>**Group Normalization (GN)** 是一种将通道（channels）分成若干组（groups）进行归一化的方法。通过调整组的数量，可以得到 Layer Normalization 和 Instance Normalization。

****

![image-20250801153930473](./assets/image-20250801153930473.png)

Problem 4(证明下降引理)

>给定函数$`f:\mathbb{R}^d \rightarrow \mathbb{R}`$的梯度是L-Lipschitz连续的,即对于任意的$`x,y \in \mathbb R^d`$,
>
>```math
>```
>
>
>
>