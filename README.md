# siamese_neural_network_OfHSV

### 中文介绍

本项目可以实现WI型OfHSV字迹鉴定
> WI型即Writer Independent，对两张签名图片判断是否为同一个人的签名
> 
> OfHSV即Offline Handwriten Signature Vertification，只使用签名照片进行判别

本项目提供了一个简单的VGG16+孪生神经网络的代码，使用pytorch

本项目还写了个网页，运行app.py后可以直接在网页上传两张签名照片，判断是否为同一个人的签名

更新：上传了数据集(CEDAR.tar.001-CEDAR.tar.011)，使用7zip解压即可。

**注意：由于签名图像的差别很小，因此可能要很长时间训练才能看出效果。数据中真实数据和虚假数据底色不一样需要预处理**

### English translation:

This project can achieve WI-type OfHSV signature identification
> WI means it can determine whether two signature images belong to the same person in a writer-independent manner. 
> 
> OfHSV refers to Offline Handwritten Signature Verification, which only uses signature photos for discrimination.

This project provides a simple code of VGG16+Siamese neural network, implemented using PyTorch.

Additionally, a web page has been created in this project. After running app.py, users can directly upload two signature photos on the webpage to determine whether they belong to the same person.

Update: The data set (CEDAR.tar.001-CEDAR.tar.011) was uploaded and unzipped using 7zip.

**Attention: Because the difference in the signature images is small, it may take a long time to train to see the effect. In the data, the background color of real data and false data is different and needs to be preprocessed**


### How to use it

1. download CEDAR dataset
2. run preprocess.py to label which pairs of images will be used
3. run train.py to train it
4. run ROC.py to judge it

Personal e-mail: 1793706453@qq.com
