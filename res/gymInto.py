#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym


# In[2]:



env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(observation, reward)

    if done:
        observation = env.reset()
env.close()


# In[5]:



env = gym.make("CartPole-v1")


# In[14]:


env.action_space.sample()


# In[9]:


env.observation_space


# In[ ]:


action_space

