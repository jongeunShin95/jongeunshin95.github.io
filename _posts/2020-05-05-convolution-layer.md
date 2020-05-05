---
title: Convolution Layer
tags: [Deep Learning, Convolution Layer]
style: fill
color: light
description: Convolution Layer 이해해보기
comments: true
use_math: true
---

<br>

#### Convolution Layer

---

Convolution Layer는 이미지와 같은 데이터에서 특징들을 추출해 내는 Layer이다.<br />
Convolution Layer를 이해하기 위해서는 filter, kernel, channel, stride, padding, activation function 등을 알아야 된다.<br>

<br>

#### Channel

---

간단하게 말해 흑백사진은 1차원으로 1개의 채널이며 컬러사진은 R, G, B를 사용하는 3차원으로 3개의 채널을 사용한다.<br />

예를 들어, (28 x 28)의 이미지가 존재할 때 흑백의 경우 shape를 사용해보면 (28, 28, 1)이 나오고 컬러의 경우 (28, 28, 3)이 된다.


<br>

#### Filter(Kernel)

---

Filter에서 해당 데이터의 특징을 추출해 낸다. 간단한 예로 3x3 input에 2x2 filter를 적용한 이미지를 나타내 보았다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/convolution_layer/filter.png?raw=true" caption="Filter Example" %}

즉, 해당 input 데이터에 대하여 filter를 적용하여 특정 값을 추출하여 새로운 Feature Map을 만드는 과정이다. 또한 kernel은 filter와 같은 의미로 쓰인다.<br />


<br>

#### Stride

---

filter를 input 데이터에 적용을 시키면서 filter가 이동될 때 얼마나 filter를 이동하는지를 나타낸다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/convolution_layer/stride_1.png?raw=true" caption="Stride 1" %}

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/convolution_layer/stride_2.png?raw=true" caption="Stride 2" %}


<br>

#### Padding

---

padding의 경우 입력 이미지에 대하여 유효한 값들만 출력 데이터로 나오게 되면서 데이터의 크기가 줄어드는 현상이 발생하는데
이 때, 입력값과 출력값의 크기가 같도록 하기 위하여 0으로 채워넣는 것을 말한다.


<br>

#### Activation Function

---

최종 출력결과를 나타내는 것으로 Filter를 통하여 나온 Feature Map에 적용하는 함수이다.
대표적인 Classification에 사용되는 소프트맥스를 예로 들면 결과 값들이 0~1 사이의 값들로 나오며 각 값들의 합은 1이 된다.
그리고 나온 값들을 one hot encoding을 적용하여 가장 큰 값에 대해 답으로 결정하게 된다. <br />

예를 들어, 0~5사이의 수를 분류하는 학습을 하여 나온 Feature Map이 다음과 같다면

* [0.1, 0.2, 0.1, 0.4, 0.1, 0.1]

one hot encoding을 적용하게 되면 [0, 0, 0, 1, 0, 0]이 되어 3이라는 숫자로 분류하게 된다.



<br>



##### 참조

---

* [https://bcho.tistory.com/1149](https://bcho.tistory.com/1149)
* [http://taewan.kim/post/cnn/](http://taewan.kim/post/cnn/)
* [https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/)
* [https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)