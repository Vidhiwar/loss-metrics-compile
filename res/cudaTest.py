#!/usr/bin/env python
# coding: utf-8

# In[47]:


import tensorflow as tf


# In[48]:


tf.test.is_gpu_available()


# In[49]:


tf.config.list_physical_devices()


# In[50]:


tf.config.list_physical_devices(
device_type=None
)


# In[ ]:





# In[39]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[52]:


import torch


# In[53]:


torch.cuda.is_available()


# In[54]:


a = torch.zeros((2,2))


# In[55]:


a = a.cuda()


# In[56]:


a


# In[57]:


tf.test.is_built_with_cuda()


# In[58]:


tf.test.gpu_device_name()


# In[ ]:





# In[ ]:




