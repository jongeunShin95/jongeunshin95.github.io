---
title: Types of Machine Learning
tags: [Machine Learning]
style: fill
color: light
description: 머신러닝의 종류
comments: true
use_math: true
---

<br>

#### Types of Machine Learning

---

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/types_of_machine_learning/types_of_machine_learning.png?raw=true" caption="Types of Machine Learning" %}

<br>

#### Supervised Learning

---

Supervised Learning에는 데이터를 학습하는데 있어서 데이터에 대한 레이블들이 존재한다.<br>
예를 들어, 고양이와 개를 분류하는 학습을 위해서는 아래의 사진과 같은 분류된 데이터들로 학습을 시키는 것이다.<br>
대표적으로 Classfication과 Regression이 있다. Classification의 경우 출력값이 이산형 변수로 나오며(개 or 고양이), Regression의 경우는 출력값이 연속형 변수로 나온다(집값 추정).

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/types_of_machine_learning/supervised_learning_example.png?raw=true" caption="Supervised Learning Example" %}

Supervisd Learning에서 사용되는 알고리즘은 다음과 같다.

* K-Nearest Neighbors
* Linear Regression
* Logistic Regression
* Support Vector Machines(SVM)
* Decision Trees
* Random Forests
* Neural Network

해당 알고리즘들에 대한 설명은 각각 포스팅으로 추가할 것이다. (공부하여 추가할 것)

<br>

#### Unsupervised Learning

---

Unsupervised Learning에는 데이터를 학습하는데 있어서 데이터에 대한 레이블들이 존재하지 않는다. 즉, 해당 데이터들의 특징들을 추출해 학습을 함므로써 데이터들을 분류하는 것이다.
대표적으로 Clustering, Dimensionality Reductino 등이 있다. 해당 학습이 사용되는 예로는 군집 분석 등이 있다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/types_of_machine_learning/unsupervised_learning_example.png?raw=true" caption="Unsupervised Learning Example" %}

Unsupervised Learning에서 사용되는 알고리즘은 다음과 같다.

* k-Means
* Hierarchical Cluster Analysis
* Principal Component Analysis
* Kernel PCA
* t-distributed Stochastic Neigbor Embedding
* Locally-Linear Embedding
* Apriori
* Eclat

해당 알고리즘들에 대한 설명은 각각 포스팅으로 추가할 것이다. (공부하여 추가할 것)

<br>

#### Semi-supervised Learning

---

Semi-supervised Learning의 경우는 레이블이 존재하는 데이터들과 존재하지 않는 데이터들을 동시에 사용하는 것이다. 대부분의 학습 데이터를 사용하게 될 경우에는 레이블이 없는 데이터들을 사용하는 경우가 더 많다. 만약 레이블이 있는 데이터를 통하여 학습을 시도하는데 이 데이터들의 양이 너무 적어 제대로 된 학습을 하지 못한는 경우에 해당 학습 기법을 이용하게 된다. 즉, 레이블이 존재하는 데이터들의 양이 너무 적을 때 레이블이 없는 데이터들을 더 추가하여 동시에 학습을 시키므로써, 조금 더 좋은 모델을 만들어 내는 것이다.

<br>

#### Reinforcement Learning

---

Reinforcement Learning의 경우에는 앞서 나온 Supervised Learning과 Unsupervised Learning과는 매우 다르다. Supervised Learning와 Unsupervised Learning의 관계는 우리가 쉽게 확인할 수 있지만, Reinforcement Learning이 포함되면 이 관계들이 살짝 애매해진다. 또한, 해당 학습 기법이 머신러닝으로 분류되지 않은 적도 있다. 앞의 두 학습 기법은 단지 데이터들을 입력하기만 하면 컴퓨터가 알아서 학습을 하여 최적의 모델을 만들어 낸다. Reinforcement Learning의 경우에는 어떠한 Action이 취해지면 그 상황에서 보상이 최대가 되도록 하는 action을 취하는 방법을 배우는 것이다. 대표적으로 알파고가 있다. 알파고는 상대의 수에 대해 수많은 가상시뮬레이션을 통하여 이길 수 있는 방법을 찾아내는 것이다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/types_of_machine_learning/reinforcement_learning_flow.png?raw=true" caption="Reinforcement Learning Flow" %}




##### 참조

---

* [https://towardsdatascience.com/what-are-the-types-of-machine-learning-e2b9e5d1756f](https://towardsdatascience.com/what-are-the-types-of-machine-learning-e2b9e5d1756f)
* [https://stickie.tistory.com/43](https://stickie.tistory.com/43)
* [https://bestpractice80.tistory.com/2](https://bestpractice80.tistory.com/2)
* [https://needjarvis.tistory.com/195](https://needjarvis.tistory.com/195)
* [https://jayhey.github.io/semi-supervised%20learning/2017/12/04/semisupervised_overview/](https://jayhey.github.io/semi-supervised%20learning/2017/12/04/semisupervised_overview/)
* [https://jaeyung1001.tistory.com/89](https://jaeyung1001.tistory.com/89)