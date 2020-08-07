第一周
* TensorFlow 概述
* TensorFlow 安装与环境配置
* TensorFlow 基础 
8 月 9 号周日下午 3-4 点，进行第一次的 Office Hours 在线解答，参与方式请关注当天早上公众号 TensorFlow（TensorFlow_official）的推送通知！

第二周
* TensorFlow 模型建立与训练 
8 月 16 号周日下午 3-4 点，进行第二次的 Office Hours 在线解答，参与方式请关注当天早上公众号 TensorFlow（TensorFlow_official）的推送通知！

第三周
* TensorFlow 常用模块
* TensorFlow 模型导出
* TensorFlow Serving 
在你和你的小队学习接近尾声时，请完成公众号 TensorFlow 将于 8 月 20 号推送的结课小测验。 谢谢。



Anaconda 环境的理解 https://www.zhihu.com/question/58033789

使用 conda 更新源修复 tensofflow 安装过程中 wrap 出现的问题 https://github.com/tensorflow/tensorflow/issues/30191  -> conda update wrapt

pip install tensorflow 需要针对虚拟环境进行设置，而不是宿主环境

理解激活函数 https://blog.csdn.net/tyhj_sf/article/details/79932893

目标/损失/代价函数区别 https://www.zhihu.com/question/52398145

2017 cs231n 课程 https://space.bilibili.com/216720985/channel/detail?cid=32406


定义： 在低维空间中不能线性分割的点集，通过转化为高维空间中的点集时，从而变为线性可分
svm 核方法 https://www.cnblogs.com/hichens/p/11874645.html

欧几里得度量无法很好衡量图像的相似性，K最近邻算法的对处理图像相似性上不可行。

矩阵相乘：前行乘后列，坐标为行列

线性分类算法 y = wx + d，其中 w 为模版分类逻辑参数，x 为输入数据源，则 y 计算得到的结果为预测结果

如何选择 w 的 值？ 

把 w 作为输入构建 loss 函数来计算衡量 w 的值好与坏，常见的有绝对值损失，log对数损失，Hinge损失等
https://baike.baidu.com/item/%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0

正则化，鼓励模型以某种方式选择更简单的w，标准的损失函数会包含 ： "数据丢失项" + "正则项"

归一化
https://houbb.github.io/2020/01/08/jieba-source-02-normalize







