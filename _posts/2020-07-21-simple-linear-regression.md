---
title: Simple Linear Regression
tags: [Deep Learning, Linear Function, regression, supervised learning]
style: fill
color: dark
description: 단순 선형 회귀에 대해 공부
date: 20-07-21 00:17:35
comments: true
use_math: true
---

<br>

#### Regression

---

[Types of Machine Learning](https://jongeunshin95.github.io/blog/types-of-machine-learning)에서 공부했던 것중 Supervised Learning에서 Classification과 Regression이 있었다. Regression은 독립 변수 X와 종속 변수 관의 선형 관계를 분석하는 것이다. 즉, Y에 대해 X가 얼마나 영향을 주는지를 분석한다.

<br>

#### Scatter plot in Simple Linear Regression

---

우선 분포되어있는 데이터에 대해 어떤 선형 분석을 해야되는지 산점도를 나타내본다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/simple_linear_regression/scatter_plot.png?raw=true" caption="scatter_plot" %}

위의 산점도를 보게 되면 우리가 최종적으로 나타내야 되는 선의 식은 다음과 같다.
<br>

> $ Y = \beta_0 + \beta_1X + \varepsilon $

여기서 $\varepsilon$은 실제의 데이터 값과 최종적으로 도출된 식의 값과의 오차를 말한다. 이 오차를 없애기 위해서는 모든 모집단 데이터에 대해 오차를 없게 만들어야 되는데 사실상 이 경우 선형으로 나오는 것이 거의 불가능하기 때문에 이 오차를 최소로 하는 식을 찾는 것이다. 우리는 위의 식을 찾기 위해 위의 모집단 데이터에서 표본을 추출하여 학습을 한 뒤 위의 식을 도출해야 된다. 여기서 우리가 예측하는 식은 다음으로 정의한다.
<br>

> $ \hat{Y} = \hat{\beta_0} + \hat{\beta_1}X$

<br>

#### SSE(Error sum of Squares)

---

SSE는 오차들의 제곱 합을 나타낸다. 식으로 나타내면

> $$ \sum_{i=1}^n e^2_i = \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_1)^2  (\because Y = \beta_0 + \beta_1X + \varepsilon) $$

과 같다. 이 SSE를 찾다가 보니 어떤 곳에서는 SSR, 또 다른 곳에서는 RSS라고 부르길래 용어가 너무 헷갈려서 따로 용어를 정리하였다. <br>

* Error sum of Squares = SSE
* Residual Sum of Squares = RSS
* Sum of Squared Residuals = SSR

또한 나중에 $ R^2 $를 공부하면서 또 다른 용어가 나오는데 헷갈리지 않기 위해 미리 정리하였다 <br>

* Regression sum of Squares = SSR
* Explained sum of Squares = SSE/ESS

그리고 위 두개를 더한 값을 나타내는 용어도 있다.

* Sum of Squares total = SST

나는 여기서 SSE(Error sum of Squares), SSR(Regression sum of Squares), SST(Sum of Squeares total)를 사용할 것이다. 표를 이용하여 나타내면 다음과 같다.

{% include elements/figure.html image="https://github.com/jongeunShin95/jongeunShin95.github.io/blob/master/assets/images/simple_linear_regression/SST_SSR_SSE.png?raw=true" caption="" %}

<br>

##### 참조

---

* [https://gentlej90.tistory.com/71](https://gentlej90.tistory.com/71)
* [https://m.blog.naver.com/PostView.nhn?blogId=istech7&logNo=50152984368&proxyReferer=https:%2F%2Fwww.google.com%2F](https://m.blog.naver.com/PostView.nhn?blogId=istech7&logNo=50152984368&proxyReferer=https:%2F%2Fwww.google.com%2F)
* [https://rk1993.tistory.com/m/entry/%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9Dlinear-regression-analysis?category=880112](https://rk1993.tistory.com/m/entry/%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9Dlinear-regression-analysis?category=880112)
* [https://365datascience.com/sum-squares/](https://365datascience.com/sum-squares/)
* [https://igija.tistory.com/256](https://igija.tistory.com/256)