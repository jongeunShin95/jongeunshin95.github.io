---
title: tf.keras.layer - Make Predictions
tags: [Deep Learning, Convolution Layer, Feature Extraction, Classification, Fully Connected, Evaluate, Predict]
style: fill
color: warning
description: 구축된 네트워크에 대한 predictions
date: 20-07-05 02:00:29
comments: true
use_math: true
---

<br>

#### Build a Network and Compile/Train

---

우선 모델에 대한 평가를 하기위하여 모델을 구축한다.

```python
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers

# Dataset
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

train_x = train_x / 255.
test_x = test_x / 255.

# Build a Network
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

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train
epochs = 1
batch_size = 32

model.fit(train_x, train_y,
         batch_size=batch_size,
         shuffle=True,
         epochs=epochs)
```

<br>

#### Make predictions

---

해당 모델에 대해 평가를 해보고 몇 개의 테스트 데이터를 직접 넣어 결과값을 확인하는 prediction을 해본다.

**Evaluate**

```python
# training 데이터가 아닌 test 데이터들을 넣어 평가한다.
model.evaluate(test_x, test_y, batch_size=batch_size)
```

    >>> [0.04268641838496551, 0.9861] // 오차율, 정확도

**Predictions**

```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# 테스트 데이터의 첫 번째를 가져온다.
image = test_x[0, :, :, 0]

# 테스트 데이터의 첫 번째 답을 시각화시킨다.
plt.title(test_y[0])
plt.imshow(image, 'gray')
plt.show()
```

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/tf_keras_layer_make-predictions/7.png?raw=true" caption="7" %}

```python
# 해당 모델에 가져온 테스트의 첫 번째 이미지를 넣는다.
# reshape를 하는 이유는 모델의 input shape를 맞추기 위해서이다.
pred = model.predict(image.reshape(1, 28, 28, 1))

print(pred)
```

    >>> array([[3.1633263e-09, 1.2009174e-07, 4.4810131e-06, 4.0004611e-06,
        4.6130970e-09, 3.3856329e-09, 2.4686831e-11, 9.9998868e-01,
        7.7280040e-09, 2.7535232e-06]], dtype=float32)
        // 총 10개의 값을 가진 배열이 나온다. -> 0~9의 값을 찾는 것이고 가장 높은 값을 가진 값이 정답으로 예측된것.

```python
print(np.argmax(pred))
```

    >>> 7 // 시각화를 통해 확인 된 값과 일치하는 것을 확인할 수 있다.
          // 즉, 해당 모델이 숫자 이미지를 정확히 맞춤.

<br>


##### 참조

---

* [https://datascienceschool.net/view-notebook/995bf4ee6c0b49f8957320f0a2fea9f0/](https://datascienceschool.net/view-notebook/995bf4ee6c0b49f8957320f0a2fea9f0/)