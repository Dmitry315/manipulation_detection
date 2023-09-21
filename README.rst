|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Детекция манипуляций в новстном потоке
    :Тип научной работы: M1P
    :Автор: Дмитрий Александрович Мелихов
    :Научный руководитель: д.ф.-м.н., Воронцов Константин Вячеславович
    :Научный консультант(при наличии): -

Abstract
========

Работа посвящена решению задачи выявления манипуляций в новостном потоке. В новостных статьях выделяются манипулятивные фрагменты и помечается тип манипуляции. Фрагменты объединяются в элементы разметки и образуют гиперграф. В работе предлагается модель на основе больших лингвистических моделей, которая выявляет фрагменты и моделирует связи между ними. Для выявления фрагментов решается задача span detection. Для постороения графа используются text2graph модель в паре с graph2text, которые обучаются c помощью техники back translation. Строится векторное представление фрагментов и предсказываются связи между ними.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
