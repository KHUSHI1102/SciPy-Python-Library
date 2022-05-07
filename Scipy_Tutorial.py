#!/usr/bin/env python
# coding: utf-8

# # Installation 

# In[ ]:


# Installing Scipy
#Windows & Linux
#using pip
#To install, run the following command in the terminal:
pip install scipy

#using Anaconda
#First Download anaconda navigator
#in anaconda prompt 
conda install -c anaconda scipy

#Mac
sudo port install py35-numpy py35-scipy py35-matplotlib py35-ipython +notebook py35-pandas py35-sympy py35-nose

#use Homebrew to install these packages
brew install numpy scipy ipython jupyter


# # Basic Functions

# # help() 

# In[40]:


#Basic Function
from scipy import cluster
help(cluster)


# # info()

# In[4]:


import scipy 
scipy.info(cluster)


# # source()

# In[5]:


scipy.source(cluster)


# # Special Functions

# # Exponential and Trigonometric Functions
# 

# In[7]:


#Special Function
#Exponential and Trigonometric Functions

from scipy import special
a = special.exp2(3)
print(a)

b = special.exp10(4)
print(b)

c = special.sindg(90)
print(c)

d = special.cosdg(60)
print(d)

e= special.tandg(45)
print(e)


# # Bessel Function 

# In[8]:


# Bessel Function of first kind of real order and complex arguement
c = special.jv(2,4)
print(c)


# # Integrate Functions

# In[11]:


# Integrate Function
from scipy import special
from scipy import integrate
i = integrate.quad(lambda x:special.exp10(x),0,1)
print(i)


# In[15]:


# Double Integrate Function
from scipy import integrate
e = lambda x, y: x*y**2
f = lambda x: 1
g = lambda x: -1
integrate.dblquad(e,0,2,f,g)


# # Linear Algebra

# In[42]:


# Linear Algebra
import numpy as np
from scipy import linalg
a = np.array([[1,2],[3,4]])
b = linalg.inv(a)
print(b)


# In[19]:


# Linear Algebra
import numpy as np
from scipy import linalg
A = np.array([[1,2],[3,4]])
B = linalg.det(A)
print(B)


# # Fourier Transform Functions

# In[43]:


from scipy.fftpack import fft, ifft
x = np.array([0,1,2,3])
y = fft(x)
print(y)


# # Waveforms

# In[45]:


from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
t = np.linspace(6, 10, 500)
w = chirp(t, f0=4, f1=2, t1=5, method='linear')
plt.plot(t, w)
plt.title("Linear Chirp")
plt.xlabel('time in sec)')
plt.show()


# # File IO

# In[46]:


import scipy.io as sio

