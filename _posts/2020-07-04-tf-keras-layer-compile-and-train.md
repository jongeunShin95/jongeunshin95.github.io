---
title: tf.keras.layer - Compile and Train
tags: [Deep Learning, Convolution Layer, Feature Extraction, Classification, Fully Connected, Compile, Train]
style: fill
color: primary
description: 구축된 네트워크에 대한 compile and train
date: 20-07-04 01:52:57
comments: true
use_math: true
---

<br>

#### Dataset and Layer 구축

---

[tf-keras-layer 구축](tf-keras-layer-build-a-network)에서 공부했던 것들을 이용하여 mnist에서 데이터셋들을 불러오고
layer층을 쌓아 하나의 모델인 네트워크를 구축한다.

**Dataset**

```python
import tensorflow as tf
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

# 차원 수 늘리기 -> 모델의 input 값이 (28, 28, 1)이기 때문.
train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

# 데이터 값 정규화
# 정규화를 하게 되면 해당 모델의 출력값이 softmax 등의 층을 지날 때는 0~1사이로 되기 때문에
# 미리 해주고 또한 1 이상의 값에 대해 다루게 될경우 값이 기하급수적으로 커지기 때문.
train_x = train_x / 255.
test_x = test_x / 255.
```

**Build a Network**

```python
from tensorflow.keras import layers

inputs = layers.Input((28, 28, 1))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net)

model.summary()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_compile_and_train/model_summary.png?raw=true" caption="model.summary()" %}

<br>

#### Compile the Model

---

컴파일하는 단계에서 필수적인 인자값으로는 Losses와 Optimizer가 있다. 또한 추가적으로 정확도를 확인하기 위해서 Metrics를 사용할 것이다.

**Losses**

손실함수는 데이터들을 학습하면서 모델에 의해 나온 출력값과 기대했던 즉, 정답인 기대값의 차이를 말한다. 최적화를 하게 되면서 이 오차를 줄여나가야 된다.
대표적으로 MSE(Mean Squared Error)가 있는데 선형회귀함수를 다루면서 해당 손실함수에 대한 포스팅을 할 것이다. 
<br><br> 우선 여기 분류에서 사용할 손실함수의 대표적인 예로는 *categorical_crossentropy*, *sparse_categorical_crossentropy*, *binary_crossentropy* 등이 있다. 
* *binary_crossentropy*의 경우 출력값이 두 개중 하나일 때 사용하는 것이고 
* *categorical_crossentropy*의 경우에는 출력값이 one-hot encoding의 형태로 나오기 때문에 우리의 모델에서의 출력값 또한 one-hot encoding으로 출력되게 해야된다.
* *sparse_categorical_crossentropy*의 경우에는 one-hot encoding이 아닌 실수의 형태로 나오기 때문에 따로 model에서 one-hot encoding을 해줄 필요가 없다.

**Metrics**

해당 Model을 통하여 나온 결과값에 대해 정확도와 오차 등 모델을 평가하는데 사용된다. 이 때 Losses 함수와 비슷하지만 이 Metrics에서의 오차는 모델을 최적화하는데 사용되지 않는다는 점이다. Losses에서 사용되는 함수들을 이용할 수도 있으며 대표적으로는 *accuracy*로 지정하여 *categorical_accuracy*를 사용한다.

**Optimizer**

손실함수로 나온 오차를 줄여가면서 모델을 최적화시키는 함수이다. 대표적으로 *경사하강법(Gradient Descent)*, *확률적 경사 하강법(Stochastic Gradient Descent, SGD)*, *아담(Adam)* 등이 있다.

**적용 예시**

```python
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

<br>

#### Train the Model

---

모델을 컴파일까지 했으면 마지막으로 해당 모델을 데이터셋을 이용하여 학습을 시키는 것이다.

**epoch과 batch size 설정**

```python
epochs = 1
batch_size = 32
```
epoch의 경우 학습을 얼마나 반복할 것인지이고, batch_size의 경우 한번 학습을 시킬 때 몇 개의 데이터들을 한 번에 학습을 시키느냐이다.

**학습 예시**

```python
model.fit(train_x, train_y,
         batch_size=batch_size,
         shuffle=True,
         epochs=epochs)
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_compile_and_train/model_training_example.png?raw=true" caption="model.fit()" %}


##### 참조

---

* [https://datascienceschool.net/view-notebook/995bf4ee6c0b49f8957320f0a2fea9f0/](https://datascienceschool.net/view-notebook/995bf4ee6c0b49f8957320f0a2fea9f0/)
* [https://keras.io/ko/](https://keras.io/ko/)
* [https://truman.tistory.com/m/164?category=854798](https://truman.tistory.com/m/164?category=854798)
* [https://tykimos.github.io/2017/09/24/Custom_Metric/](https://tykimos.github.io/2017/09/24/Custom_Metric/)
* [https://kevinthegrey.tistory.com/118](https://kevinthegrey.tistory.com/118)
* [https://www.tensorflow.org/tutorials/keras/classification?hl=ko](https://www.tensorflow.org/tutorials/keras/classification?hl=ko)