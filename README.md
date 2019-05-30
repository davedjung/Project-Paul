# Project Paul: Maxwell-Boltzmann Distribution Simulation

created by Jung Min Ki

## Introduction

Project Paul is essentially a particle simulation designed for the sole purpose of experimentally verifying the Maxwell-Boltzmann distribution.<br />

## User Guide

### I. Run the simulation

If you have any experience in coding in Python, you could just simply run **simulation.py**. You will need to install these packages [It is recommended to use **pip** to install these packages]:
1. NumPy (to learn more about NumPy, visit: https://www.numpy.org/)
2. Matplotlib (to learn more about Matplotlib, visit: https://matplotlib.org/)

If you have no experience with Python, download the entire **Release** folder and run the following file:
> /Release/dist/simulation/simulation.exe

### II. Input parameters
1. The first parameter that needs to be specified is the number of particles to simulate. Ideally, that number is a positive integer (around **1000** particles are recommended).
2. The second parameter is dubbed "Temperature Scale," even though technically it is not. This so-called scale determines the initial velocity distribution, which is related to the temperature of the system. This can be any positive value (For optimal effects, input value of **2** is recommended).
3. The third parameter specifies the mode of computation; that is, whether to prioritize accuracy or speed.
   - **Mode 0** is physically accurate, but is quite slow (Even though it is approximately 100 times faster than version alpha). It prioritizes **precision**.
   - **Mode 1** is also physically accurate when handling each collision, but ignores some collisions completely. However, the results are still valid. It prioritizes **performance**.

## Links to YouTube videos

1. Precision Mode https://youtu.be/DsA8kGd44UA<br />
Number of particles: 1000<br />
Temperature scale: 2<br />
100x faster than version alpha, but real-time simulation is still unattainable. Time-lapse video of 12-hour-long screen capture footage.

2. Performance Mode https://youtu.be/DsA8kGd44UA<br />
Number of particles: 1000<br />
Temperature scale: 2<br />
Accurate calculation in terms of physics, but a significant portion of collisions is ignored. Real-time simulation achieved.
