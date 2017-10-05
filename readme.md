Recently, I came across a paper called [Random feedback weights support learning
in deep neural networks](https://arxiv.org/abs/1411.0247). 
I found it very fascinating that they were able to match the performance of backpropogation algorithm by propagating the error through random weights! (instead of the same weights they used for forward prop)
 Yes random weights! The idea is inspired by the biological limitation of the brain to deliver exact backpropogation information. 
 
 I decided to replicate this paper and take a closer look for myself. 