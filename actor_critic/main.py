import sys
import gym
import numpy as np
import tensorflow as tf 
from ac import ActorCritic 

N_epochs = 500
min_epochs = 200
T_steps = 20

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n 

    model = ActorCritic(num_actions)

    ep_rewards = 0.0
    best_reward_so_far = 0.0
    for epoch in range(N_epochs):
        s = env.reset()
        done = False 
        
        states = []
        next_states = [] 
        actions = []
        rewards = [] 
        dones = [] 
        steps = 0 
        while not done : 
           
            prob, _ = model(s[None,:])
            action = np.random.choice(num_actions, p=np.squeeze(prob))
            action_prob = prob[0, action]
            s_prime, rwd, done, _ = env.step(action)
            ep_rewards += rwd

            states.append(s)
            actions.append(action)
            rewards.append(rwd)
            next_states.append(s_prime)
            dones.append(done)

            if steps == T_steps or done : 
                with tf.GradientTape() as tape : 
                    loss = model.compute_loss(states, [s_prime], actions, rewards, dones)

                gradient = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

                states = []
                next_states = [] 
                actions = []
                rewards = [] 
                dones = [] 
                steps = 0  
                
            s = s_prime
            steps += 1
        
        # if epoch > min_epochs and best_reward_so_far < ep_rewards :
        #     best_reward_so_far = ep_rewards
        #     model.save('ac_model')
        #     print("model saved")
        
        if (epoch+1) % 20 == 0 : 
            print ("Epoch [%d/%d] : Reward %d"%(epoch + 1, N_epochs, ep_rewards/20))
            ep_rewards = 0.0 


def test():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n  
    model = tf.keras.models.load_model('ac_model')
      
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
