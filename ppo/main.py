import sys
import gym
import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp

# tf.keras.backend.set_floatx('float64')

class Memory : 
    def __init__(self):
        self.actions = [] 
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.values = [] 
        self.rewards = [] 
        self.dones = [] 
    
    def clear_memory(self):
        self.actions = [] 
        self.states = []
        self.next_states = [] 
        self.logprobs = []
        self.values = [] 
        self.rewards = [] 
        self.dones = [] 


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        initializer = 'random_normal'
        self.fc_pi1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)
        self.fc_pi2 = tf.keras.layers.Dense(num_actions, activation='softmax', kernel_initializer=initializer)
        self.fc_val1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)
        self.fc_val2 = tf.keras.layers.Dense(1, kernel_initializer=initializer)

    
    # @tf.function
    def call(self, states):
        pi = self.fc_pi1(states)
        pi = self.fc_pi2(pi)
        val = self.fc_val1(states)
        val = self.fc_val2(val)

        return pi, val


class PPO : 
    def __init__(self, num_actions, N_iter, gamma, lmbda, eps_clip, lr):
        self.policy = ActorCritic(num_actions)
        self.old_policy = ActorCritic(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.N_iter = N_iter
        self.gamma = gamma 
        self.lmbda = lmbda
        self.eps_clip = eps_clip

    def get_dist(self, logits):
        dist = tfp.distributions.Categorical(probs=logits)
        return dist 

    def update(self, memory):

        old_states = tf.convert_to_tensor(memory.states, dtype=tf.float32)
        old_actions = tf.convert_to_tensor(memory.actions, dtype=tf.int32)
        old_logprobs = tf.convert_to_tensor(memory.logprobs, dtype=tf.float32)
        next_states = tf.convert_to_tensor(memory.next_states, dtype=tf.float32)
        
        dones = tf.convert_to_tensor(np.array(memory.dones) * 1.0 , dtype=tf.float32)

        for i in range(self.N_iter):
            with tf.GradientTape() as tape:
                # Run the new policy with old and new states
                pi, old_values  = self.policy(old_states)
                _, next_values = self.policy(next_states)

                # =================
                # Compute advantage 
                # =================
                old_values = tf.squeeze(old_values)
                next_values = tf.squeeze(next_values)

                td_target = memory.rewards + self.gamma * next_values * (1.0-dones)
                delta = tf.stop_gradient(td_target - old_values) 

                advantages = []
                adv = 0.0 
                for d in reversed(delta): 
                    adv = d + adv * self.gamma * self.lmbda 
                    advantages.append(adv)
                advantages = advantages[::-1]
                advantages = tf.convert_to_tensor(advantages)

                # ===========================
                # compute clipped policy loss
                # ===========================
                dist = self.get_dist(pi)
                log_probs = dist.log_prob(old_actions) 
                entropy = dist.entropy()
                
                ratio = tf.math.exp(log_probs - old_logprobs)
                
                advantages = tf.stop_gradient(advantages)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = tf.reduce_mean(tf.math.minimum(surr1, surr2)) 

                # ===================
                # compute value loss
                # ===================
                value_loss = tf.reduce_mean(tf.math.square(td_target - old_values))

                loss = - policy_loss + value_loss * 0.5 + tf.reduce_mean(entropy) * 0.01

            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        
        self.old_policy.set_weights(self.policy.get_weights())

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n 

    N_epochs = 5000          # total number of episodes to train with
    min_epochs = 200         # 
    update_steps = 64        # maximum number of steps to sample 
    N_iter = 3               # number of iterations for each trajectory 
    gamma = 0.99             # discount factor
    lmbda = 0.95             # GAE factor 
    eps_clip = 0.2           # clipping value for PPO loss
    lr = 0.0005              # learning rate for optimizer 

    ppo = PPO(num_actions, N_iter, gamma, lmbda, eps_clip, lr)
    memory = Memory()

    total_reward = 0.0
    
    for epoch in range(1, N_epochs+1):
        s = env.reset()
        done = False
        while not done : 
            for e in range(update_steps):  
                # run the policy and save to memory 
                s_ = tf.expand_dims(tf.convert_to_tensor(s), 0)
                pi, val = ppo.old_policy(s_)

                dist = ppo.get_dist(pi)
                action = dist.sample()
                log_prob = dist.log_prob(action) 
                # action = np.random.choice(num_actions, p=np.squeeze(prob))
                # action_prob = prob[0, action]
                action = tf.squeeze(action).numpy()
                log_prob = tf.squeeze(log_prob).numpy()
                val = tf.squeeze(val).numpy()

                memory.actions.append(action)
                memory.states.append(s)
                memory.logprobs.append(log_prob)
                memory.values.append(val)

                s, rwd, done, _ = env.step(action)

                memory.rewards.append(rwd)
                memory.dones.append(done)
                memory.next_states.append(s)
                total_reward += rwd
                
                if done : 
                    break 

            ppo.update(memory)
            memory.clear_memory()
        
        if epoch % 20 == 0 : 
            print("Epoch %d/%d : reward %d" % (epoch, N_epochs, total_reward/20))
            total_reward = 0 


    

def test():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n  
    model = tf.keras.models.load_model('ppo_model')
      
    done = False 
    s = env.reset() 
    rewards = 0 
    while not done : 
        env.render()
        probs, _ = model(s[None, :])
        action = tf.math.argmax(probs, axis=1).numpy()[0]
        s, rwd, done, _ = env.step(action)
        rewards += rwd
    print(rewards)


if __name__ == '__main__': 
    if len(sys.argv) != 2 : 
        print ("check argument")
    elif sys.argv[1] == 'train': 
        main()
    elif sys.argv[1] == 'test' : 
        test()
