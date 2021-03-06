---
title: tf.keras.layer - Build a Network
tags: [Deep Learning, Convolution Layer, Feature Extraction, Classification, Fully Connected]
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

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_build_a_network/x_train.png?raw=true" caption="x_train[0] 이미지" %}


**y_train의 데이터 보기**

```python
result = y_train[0]
print(result)
```
    >>> 5


보게 되면 x_train의 데이터에는 학습을 통하여 우리가 맞추어야 되는 문제들이 저장되어 있고, y_train의 경우에는 그 x_train의 답이 저장되어 있다.<br>
이 mnist와 keras를 이용하여 데이터를 학습하기 위한 layer를 구축해 볼 것이다.
<br>

#### tf.keras.layers.Conv2D

---

저번에 Convolution Layer에서 본 것처럼 해당 데이터들에서 특징을 추출하는 layer이다.<br>
[Tensorflow 문서](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)에서 보게되면 여러 파라미터들이 존재한다. 여기서 filter, kernel_size, strides, padding, activation에 대해 설정하여 사용해 볼 것이다. 여기서 파라미터에 대한 설명은 [Convolution Layer](https://jongeunshin95.github.io/blog/convolution-layer)에서 다루었고 filters의 파라미터에 대해서만 다르게 적용하여 사용해 볼 것이다.

**기본적인 layer를 구성하는 코드**

```python
tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')
```
filters의 경우 해당 layer를 통과한 후 몇개의 filter를 만드는지 결정한다. 한번 filters를 다르게 하여 layer를 통과시킨 후 결과를 보겠다. 또한 다음 예제들을 실행할 때 activation은 설정하지 않고 activation의 경우에는 따로 설정을 하도록 하겠다.

**layer에 사용될 데이터 설정**

```python
image = train_x[0]  # 학습용 데이터에서 첫 번째 데이터를 가지고온다.
image = image[tf.newaxis, ..., tf.newaxis] # 필터 수와 채널 수를 설정해준다. (28, 28) -> (1, 28, 28, 1)
image = tf.cast(image, dtype=tf.float32) # 데이터 타입을 변경해준다.
image.shape
```

    >>> (1, 28, 28, 1)

**filters=3인 layer**

```python
layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
output = layer(image)
output.shape
```

    >>> TensorShape([1, 28, 28, 3])

**filters=5인 layer**

```python
layer = tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
output = layer(image)
output.shape
```

    >>> TensorShape([1, 28, 28, 5])

두 결과를 보게되면 layer를 통과한 데이터로 나오는 filter의 수가 3, 5로 되는 것을 볼 수 있다. 즉 kernel_size=(3, 3)인 3x3의 filter를 통과하는데 총 몇개의 filter를 통과하여 몇개의 결과로 나오는지를 정하는 것이다. <br>

**layer를 통과한 후 이미지 시각화(filters=5)**

```python
plt.subplot(1, 2, 1)
plt.imshow(image[0, :, :, 0], 'gray')
plt.subplot(1, 2, 2)
plt.imshow(output[0, :, :, 0], 'gray')
plt.show()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_build_a_network/layer_apply_image.png?raw=true" caption="layer 적용 전 / 적용 후" %}

<br>

#### tf.keras.layers.ReLU

---

나오는 결과에 대하여 0보다 작은 경우에는 모두 0으로 만드는 Activation Function이다.

**기본적인 형태**

```python
tf.keras.layers.ReLU()
```

위의 layer를 통하여 나온 output에 대하여 Relu함수를 적용시켜보겠다.

**Relu 적용시키기 전**

```python
import numpy as np
print(np.min(output), np.max(output))
```

    >>> (-312.53235, 187.89764)

**Relu 적용시킨 후**
```python
relu = tf.keras.layers.ReLU()
relu_output = relu(output)
print(np.min(relu_output), np.max(relu_output))
```

    >>> (0.0, 187.89764) # 0 보다 작은 수는 다 0으로 치환되었다.

**적용시킨 후 이미지 시각화**
```python
plt.subplot(1, 2, 2)
plt.imshow(output[0, :, :, 0], 'gray')

plt.subplot(1, 2, 1)
plt.imshow(relu_output[0, :, :, 0], 'gray')
plt.show()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_build_a_network/Relu_output.png?raw=true" caption="ReLU 적용 전 / 적용 후" %}

<br>

#### tf.keras.layers.MaxPool2D

---

Pooling의 경우 해당 데이터가 주어졌을 때 해당 구간에서 가장 큰 값들을 가져와서 합쳐놓는 것이다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_build_a_network/pooling.png?raw=true" caption="Pooling" %}

즉, 정해진 구간 내에서 가장 큰값들을 가져오는 함수이다.

**기본 형태**
```python
tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
```
여기서 pool_size가 2x2만큼의 크기를 정해주는 것이고 strides는 얼만큼 그 구간을 옮겨 반복하는지를 정한다. strides의 경우 [Convolution Layer](https://jongeunshin95.github.io/blog/convolution-layer)에서도 다루었다.

**MaxPool2D 적용 전**
```python
print(relu_output.shape)
```

    >>> TensorShape([1, 28, 28, 3])

**MaxPool2D 적용 후**
```python
pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
pool_output = pool_layer(relu_output)
print(relu_output.shape)
```

    >>> TensorShape([1, 14, 14, 3])

해당 데이터에서 큰 값들만 가져오기 때문에 shape의 크기가 반으로 줄어든 것을 볼 수 있다.

**적용시킨 후 이미지 시각화**
```python
plt.subplot(1, 2, 1)
plt.imshow(relu_output[0, :, :, 0], 'gray')

plt.subplot(1, 2, 2)
plt.imshow(pool_output[0, :, :, 0], 'gray')
plt.show()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_build_a_network/pool_output.png?raw=true" caption="MaxPool2D 적용 전 / 적용 후" %}

<br>

#### tf.keras.layers.Flatten

---

flatten의 경우에는 해당 n x m 의 데이터를 열벡터 형식으로 나열하는 것이다.

**적용시키기 전 shape**
```python
print(pool_output.shape)
```

    >>> TensorShape([1, 14, 14, 3])

**적용시킨 후 shape**
```python
layer = tf.keras.layers.Flatten()
flatten = layer(pool_output)
print(flatten.shape)
```

    >>> TensorShape([1, 588])

보게 되면 TensorShape([1, 14, 14, 3])에서 14 * 14 * 3 = 588로 제대로 나오는 것을 확인할 수 있다.

<br>

#### tf.keras.layers.Dense

---

Dense의 경우에는 입출력에 관해 모두 연결을 해준다. 또한 이 연결에 있어서 가중치가 있는데 가중치의 크기에 따라 입력 노드에 대해 출력 노드에 미치는 영향이 다르다.

**Dense Layer 적용**
```python
layer = tf.keras.layers.Dense(32, activation='relu')
dense = layer(flatten)
print(dense.shape)
```

    >>> TensorShape([1, 32])

출력 결과를 보게되면 Flatten에서 나온 입력 588개에 대하여 32개의 출력으로 모두 연결을 시켰다.

<br>

#### tf.keras.Dropout

---

학습 데이터에 대해 너무 집중이 되다 보면 학습 데이터들은 잘 맞추지만 실제 데이터에 대해서는 못맞추는 경우가 발생하는데 이것이 Overfitting이다. 이 Overfitting을 방지하기 위하여 연결된 모든 학습 데이터에 대해 학습을 시키는 것이 아니라 몇 개의 연결을 끊어서 조금 더 낮은 가중치에 대한 연결에도 학습을 할 수 있게 하는 것이다.

**Dropout 사용예시**
```python
layer = tf.keras.layers.Dropout(0.7)
dropout = layer(dense)
```

<br>

##### 참조

---

* [https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/)
* [https://tykimos.github.io/2017/01/27/Keras_Talk/](https://tykimos.github.io/2017/01/27/Keras_Talk/)
* [https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/](https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/)