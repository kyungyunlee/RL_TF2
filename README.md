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
* REINFORCE with baseline
* Actor critic

* PPO 

### References 
