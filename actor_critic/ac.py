import sys
import tensorflow as tf 
import numpy as np



class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.fc_pi1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc_pi2 = tf.keras.layers.Dense(num_actions, activation='softmax')
        self.fc_val1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc_val2 = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(0.005)

    @tf.function
    def call(self, states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        pi = self.fc_pi1(states)
        pi = self.fc_pi2(pi)
        val = self.fc_val1(states)
        val = self.fc_val2(val)
        return pi, val

    def compute_loss(self, states, next_states, actions, rewards, dones, discount_factor=0.99):
        probs, values = self.call(states)
        actions = np.expand_dims(actions, 1)
        action_probs = tf.gather_nd(probs, actions, batch_dims=1)

        _, next_values = self.call(next_states)
        
        values = tf.squeeze(values)
        next_values = tf.squeeze(next_values)
        # compute advantage 
        R = next_values
        returns = [] 
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + discount_factor * R * (1-done)  
            returns.append(R)
        returns = returns[::-1]

        adv = returns - values
        actor_loss = - tf.multiply(tf.math.log(action_probs), tf.stop_gradient(adv))
        critic_loss = tf.math.square(adv)
        loss = tf.reduce_mean(actor_loss + critic_loss )
        return loss 

