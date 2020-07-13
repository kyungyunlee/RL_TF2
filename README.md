# Reinforcement learning algorithms in TF 2

### Algorithms
* REINFORCE - monte-carlo policy gradient 
$$ \nabla_\theta J(\theta) = \mathbb{E}[G_t \nabla log \pi(A_t|S_t)]$$ 
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
* learning rate seems to matter a lot 
* tf.reduce_sum vs tf.reduce_mean can have an effect 


### References 
