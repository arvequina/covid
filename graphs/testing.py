#!/usr/bin/python
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def new_function(input):
    if input < 100:
        print(input)
    else:
        #add 100 to the result
        input = input + 100
        print (input)

new_function(110)
new_function(9)