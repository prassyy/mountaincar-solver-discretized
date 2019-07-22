# Mountain Car Solver

This project is aimed to solve the MountainCar-v0 environment in OpenAI Gym. The task is to learn to ride a car to the top of mountain against gravity by gaining adequate momentum.

<img src="/assets/mountain-car-converged.gif" height=480/>

The observation space is the position and velocity of the car at any moment. The continuous state space is discretised into blocks of 10x10.

The agent is rewarded **-1** for every second of the episode!
The episode ends when the car reaches the top of the mountain.

