![DQN-Snake](./img/logo3.png)

# Play Snake with Deep Q-Learning
#### by Jae-Hyeong, Sim([@Kuax-Meat](https://github.com/Kuax-Meat/))
If you have any question, suggest an issue or send an email to me. `mirelurk at kuax.org`

## 1. Introduction
This repo is an agent that plays `snake` game with Google DeepMind's DQN(NIPS 2013).

## 2. Dependencies
#### Fully compatible with
+ Windows 10 Professional 64-bit
+ Anaconda 4.2.0 64-bit
+ Tensorflow RC 1.0
+ Pygame

## 3. ConvNet
Uses 3 hidden Convolutional NN layer(same as DQN Nature 2015).

## 4. How to run
```
> python train.py
```

## 5. Result
Details of `Snake` for 5 hours(775,136 Frames and 63,033 Episodes) with Nvidia Geforce GTX 1070
![QValue](./img/avg_qv.jpg)
![AvgScore](./img/avg_score.jpg)

## 6. Disclaimer
This repo is highly based on

+ https://github.com/devsisters/DQN-tensorflow
+ https://github.com/asrivat1/DeepLearningVideoGames

and yes, Google DeepMind's DQN. https://deepmind.com/research/dqn/

Game `Snaky` Raw Code from
https://inventwithpython.com/pygame/chapter6.html

Thanks to Sung Kim([@hunkim](https://github.com/hunkim/))

http://hunkim.github.io/ml/