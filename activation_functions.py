# Skok jednostkowy
def step_response(s):
    if s < 0:
        return 0
    
    return 1

# Signum
def signum(s):
    if s < 0:
        return -1
    
    return 1

# Odcinkowo liniowa
def line_segment(s, c=1):
    if s < 0:
        return 0
    
    if s < c:
        return s
    
    return c

# Liniowa
def linear(s, a=1, b=0):
    return s * a + b

# Sigmoidalna
def sigmoid(s):
    import numpy as np

    y = 1 + np.exp(-s * 2)
    
    return (1 / y)

def deriv_sigmoid(s):
    v = sigmoid(s)
    return v * (1 - v)

# Tangens hiperboliczny
def tanh(s):
    import math
    
    x = 1 - math.e ** (-s)
    y = 1 + math.e ** (-s)
    
    return (x / y)

def deriv_tanh(s):
    v = tanh(s)

    return (1 - v * v)

def random_values(n=1):
    v = []

    for i in range(0, n):
        v.append(random())
    
    return v
