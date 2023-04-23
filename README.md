# statistical_arbitrage

## Overview

A pairs trading strategy is one of the most popular trading strategies when it comes to finding trading opportunities between two stocks that are co-integrated. 
The pairs trading strategy assumes the highly correlated securities will come back to their neutral position after any divergence. This strategy can be incorporated into any kind of trading and in any market such as stocks, forex, etc. 

## Methodology

<img width="658" alt="Monosnap PowerPoint | Microsoft Teams 2023-04-23 16-08-38" src="https://user-images.githubusercontent.com/114669230/233847787-1021f581-4d8d-45d2-973c-9b331416073e.png">


1. Normalisation: Normalise prices during the formation and trading period.
2. Pairs Generation: Calculate the squared differences among all possible combinations of pairs and add them up. Select the **10 pairs** with the lowest sum of their squared differences.
3. Signal generation: Compute statistics of differences between the normalised prices of all possible combinations of pairs during the formation period (mean and standard deviation). A signal to open a position in the trading period is triggered when the actual difference is the mean plus two standard deviations. A signal to close an open position is triggered when the actual difference converges back to the mean.
4. Trading period: At the end of every trading day it is calculated the actual spread between the normalised prices of each of the 10 pairs found in step 2. The actual spread is compared with the signals. If there is a signal to open a position at the end of one day, it is assumed that one can open the position using the opening price the next day. 

## Results

**Value portfolio at the end of period (04/2022) - Starting value 1 (01-1996):**

<img width="160" alt="Monosnap benchmark_st… (16) - JupyterLab 2023-04-23 16-12-46" src="https://user-images.githubusercontent.com/114669230/233847987-c96f139d-ca01-40e2-924a-0381d7f627a3.png">

**Note:** The number 1 refers to the most cointegrated pairs. The number 10 is the least cointegrated pair from the top 10 most cointegrated pairs.

<img width="852" alt="Monosnap benchmark_st… (16) - JupyterLab 2023-04-23 16-11-25" src="https://user-images.githubusercontent.com/114669230/233847927-70a35e80-0d7f-4f56-9014-434a8ff278b2.png">

**Summary Statistics:**

<img width="917" alt="Monosnap benchmark_st… (16) - JupyterLab 2023-04-23 16-15-05" src="https://user-images.githubusercontent.com/114669230/233848107-ba6ed1a3-ab10-4c9a-90a3-6b7860c0cb9a.png">

## Main Findings

* Due to a more competitive environment, the strategy returns have been decreasing over the years.
* Different versions of pairs trading behave poorly over the most liquid stocks. However, by adding more "exotic" stocks to the analysis the returns from the backtesting can increase dramatically. The question now becomes on the execution in real-time of those trades with more "exotic" stocks. Is it better to tweak the strategy to find a better performance or to focus on finding an alternative dataset where the strategy still performs well?
