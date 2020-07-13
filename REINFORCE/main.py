import os
import sys
import gym
import numpy as np
import tensorflow as tf 
from reinforce import Reinforce 

N_epochs = 400

def generate_trajectory(env, model):
    states = []
    actions = []
    rewards = [] 

    s = env.reset()
    done = False 
    while not done :
        states.append(s)
        probs = model(s[None,:])
        action = tf.squeeze(tf.random.categorical(tf.math.log(probs), 1), axis=-1).numpy()[0]
        s, rwd, done, _ = env.step(action)
        actions.append(action) 
        rewards.append(rwd)
    
    return states, actions, rewards 


def test() : 
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n  
    model = tf.keras.models.load_model('reinforce_model')
    
    done = False 
    s = env.reset() 
    rewards = 0 
    while not done : 
        env.render()
        probs = model(s[None, :])
        action = tf.math.argmax(probs, axis=1).numpy()[0]
        s, rwd, done, _ = env.step(action)
        rewards += rwd
    print(rewards)



def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n 

    model = Reinforce(state_size, num_actions)

    for epoch in range(N_epochs): 
        with tf.GradientTape() as tape : 
            # sample trajectory
            states, actions, rewards = generate_trajectory(env, model)
            discounted_rewards = model.compute_discount(rewards)
            loss = model.loss(states, actions, discounted_rewards)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch %20 == 0 : 
            print ("Epoch [%d/%d] : Reward %d"%(epoch, N_epochs, np.sum(rewards)))

    model.save('reinforce_model')
    print("model saved!")
    

if __name__ == '__main__': 
    if len(sys.argv) != 2 : 
        print ("check argument")
    elif sys.argv[1] == 'train': 
        main()
    elif sys.argv[1] == 'test' : 
        test()
