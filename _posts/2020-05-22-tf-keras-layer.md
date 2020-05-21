---
title: tf.keras layer
tags: [Deep Learning, Convolution Layer, Feature Extraction, Classification]
style: fill
color: success
description: tf.keras를 이용한 layer 구축
date: 20-05-22 00:42:09
comments: true
use_math: true
---

<br>

#### datasets - mnist

---

저번에 convolution layer에서 공부했던 특징들을 추출하는 부분을 tf.keras를 이용하여 간단하게 구현해보기로 했다. <br>우선 사용할 데이터로는 keras에서 제공해주는 datasets의 mnist를 이용하기로 하였다. minist는 숫자가 적힌 그림이 있고 또한 그 그림에 해당하는 숫자가 정답으로 있는 데이터셋이다. 우선 어떤 데이터인지 간단하게 시각적으로 확인을 해보자.

**필요한 라이브러리들 불러오기**

```python
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
%matplotlib inline
```
mnist를 사용하기 위해 datasets을 불러오고 시각화를 위해 matplotlib을 이용한다.

**데이터 받아오기**

```python
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
```
데이터를 받아오는데 학습용 데이터 밑 테스트 용 데이터들을 받아온다.<br>
x의 경우에는 숫자가 그려진 그림이 나오고 y의 경우에는 그 사진의 숫자가 저장되어있다.

**x_train의 데이터 보기**

```python
image = x_train[0]
plt.imshow(image, 'gray')
plt.show()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/keras_layer/x_train.png?raw=true" caption="x_train[0] 이미지" %}


**y_train의 데이터 보기**

```python
result = y_train[0]
print(result)
```
    >>> 5


보게 되면 x_train의 데이터에는 학습을 통하여 우리가 맞추어야 되는 문제들이 저장되어 있고, y_train의 경우에는 그 x_train의 답이 저장되어 있다.<br>
이 mnist와 keras를 이용하여 데이터를 학습하기 위한 layer를 구축해 볼 것이다.
<br>

#### 입력

---



<br>

##### 참조

---

* [https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/)
* [https://tykimos.github.io/2017/01/27/Keras_Talk/](https://tykimos.github.io/2017/01/27/Keras_Talk/)