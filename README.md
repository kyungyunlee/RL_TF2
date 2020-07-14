# Reinforcement learning algorithms in TF 2
**WIP**
### Algorithms
* REINFORCE - monte-carlo policy gradient 
```
initialize policy parameters
for N epochs 
    sample an episode using the current policy (save reward, action, state)
    compute the discounted reward for each step 
    for each step 
        compute loss and update parameters
```
  * Need to sample full trajectory to update -> even bad actions are considered good if the total reward is high 
* REINFORCE with baseline
* Actor critic - TD learning method 
  * If doing TD actor-critic, T = 1 ? 
```
initialize policy and value function parameters
for N epochs
    while not done 
        for T steps 
            sample s,a 
        compute policy and actor loss 
        update parameters
```

* PPO 

### Comments 
* So many parameters to tune...
    * learning rate seems to matter quite a lot 
    * Initialization also important
* tf.reduce_sum vs tf.reduce_mean can have an effect 
* What is a good way for finding an optimal model? 
* How to optimize RL codes?

### References 
* [https://github.com/seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)
* [https://github.com/nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
