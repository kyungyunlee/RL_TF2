import sys
import tensorflow as tf
import numpy as np 

class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(Reinforce, self).__init__()
        self.num_actions = num_actions

        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_actions, activation='softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    @tf.function
    def call(self, states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        x = self.fc1(states)
        x = self.fc2(x)
        return x 

    def compute_discount(self, rewards, discount_factor=.99):
        R = 0
        discounted_rewards = []

        for r in reversed(rewards):
            R = R * discount_factor + r 
            discounted_rewards.append(R)
        discounted_rewards = discounted_rewards[::-1]
        return discounted_rewards


    def loss(self, states, actions, discounted_rewards):
        
        probs = self.call(states)
        actions = np.expand_dims(actions, 1)
        action_probs = tf.gather_nd(probs, actions, batch_dims=1)
        loss = tf.math.multiply(tf.math.log(action_probs), discounted_rewards) 
        loss = -tf.reduce_sum(loss)
        return loss 

