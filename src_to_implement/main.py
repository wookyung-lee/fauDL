import numpy as np
import matplotlib.pyplot as plt

from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator


part = input("Please enter the number of exercise you want to run: 1.2, 1.3, 1.4 or 2. ")

valid_parts = ["1.2", "1.3", "1.4", "2"]
while part not in valid_parts:
    part = input("Please choose between 1.2, 1.3, 1.4, 2: ")

if part == "1.2":
    tile_size_1 = int(input("tile_size for Checker: "))
    resolution_1 = int(input("resolution for Checker: "))
    Q12 = Checker(resolution_1,tile_size_1)

    if resolution_1 % (tile_size_1*2) != 0:
        print("Invalid values entered, try again")
    else:  
        Q12.show()
        plt.show()
        
elif part == "1.3":
    resolution_2 = int(input("resolution for Circle: "))
    radius = int(input("radius for Circle: "))
    position = input("position for Circle, enter as x,y: ")

    pos = tuple(map(int, position.split(',')))

    Q13 = Circle(resolution_2, radius, pos)
    Q13.show()
    
elif part == "1.4":
    resolution_3 = int(input("resolution for Spectrum: "))
    Q14 = Spectrum(resolution_3)
    Q14.show()

elif part == "2": 
    x = ImageGenerator("exercise_data", "Labels.json", 10, (32,32,3), True, True, True)
    x.show()
    