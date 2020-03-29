---
title: Generator
tags: [Javascript, Generator, JS]
style: 
color: 
description: Javascript - Generator
comments: true
---

## 1. Generator?

현재 Javascript에서 비동기를 다룰 때 Async/await를 많이 사용하는데 예전에 비동기를 다룰 때 사용했는 문법이다. (조금 더 알아봐야될듯)

## 2. Usage

### 2-1. 값을 전달

```javascript
function* setValue() {
    console.log(yield);
    console.log(yield);
    console.log(yield);
}

const test = setValue()

test.next();
test.next(1);
test.next(2);
test.next(3);
```

    >>> 1
        2
        3

### 2-2. 값을 받기

```javascript
function* getValue() {
    yield 1
    yield 2
    yield 3
  }
  
  for (i of getValue()) { 
    console.log(i)
  }
```

    >>> 1
        2
        3