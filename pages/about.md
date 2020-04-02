---
layout: page
title: About
permalink: /about/
weight: 3
---

# **About Me**

Hi I am **{{ site.author.name }}** :wave:,<br>
I am studying JS, C/C++, Python, Deep Learning, Mathematics...<br>
<!-- Go site -> [mathematics site](https://www.notion.so/5e0a5b037dfc44bbaae8eec5cc870661?v=ef7edc0d640c47a9acf84b944e487d67) -->

<div class="row">
{% include about/skills.html title="Programming Skills" source=site.data.programming-skills %}
{% include about/skills.html title="Other Skills" source=site.data.other-skills %}
</div>

<div class="row">
{% include about/timeline.html %}
</div>