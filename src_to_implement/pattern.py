import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tsize):
        if resolution%(2*tsize) != 0:
            raise ValueError()
        self.tsize = tsize
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution), dtype=int)
        num_tiles = self.resolution//self.tsize
        for x in range(num_tiles):
            for y in range(num_tiles):
                if (x+y)%2 != 0:
                    current_x = x*self.tsize
                    current_y = y*self.tsize
                    for i in range(self.tsize):
                        self.output[current_y+i, current_x:current_x+self.tsize] = 1
        return self.output.copy()
    
    def show(self): 
        output = self.draw()
        plt.imshow(output, cmap='gray')        
        plt.axis('off')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        y, x = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution), indexing='ij')
        center_x, center_y = self.position
        distance = np.sqrt((x-center_x)**2 + (y-center_y)**2)
        self.output = np.zeros((self.resolution, self.resolution), dtype=int)
        self.output[distance <= self.radius] = 1
        return self.output.copy()

    def show(self):
        output = self.draw()
        plt.imshow(output, cmap='gray')
        plt.axis('off')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
    
    def draw(self):
        """
        We need Matrix (array of arrays) of tuples:
        [
        [(0, 0, 255), (1, 0, 254), ... , (255, 0, 0)],
        [(0, 1, 255), (1, 0, 254), ... , (255, 1, 0)],
        ...
        [(0, 255, 255), (1, 255, 254), ... , (255, 255, 0)]
        ]

        We can see that:
          the red channel in the rows has always (0, 1, ..., 255) pattern
          the green channel in the rows has always (0, 0, ..., 0), (1, 1, ..., 1), ... ,(255, 255, ..., 255) pattern
          the blue channel in the rows has always (255, 254, ..., 0) pattern        
        
          We use linspace to obtain given resolution
        """
        sequence = np.linspace(0, 1, self.resolution)
        red = np.tile(sequence, (self.resolution, 1))                     
        green = np.tile(sequence.reshape(-1, 1), (1, self.resolution))

        rev_sequence = np.linspace(1, 0, self.resolution)
        blue = np.tile(rev_sequence, (self.resolution, 1)) 
        self.output = np.stack((red, green, blue), axis=-1)
        return self.output.copy()

    def show(self):
        output = self.draw()
        plt.imshow(output, cmap='gray')
        plt.axis('off')
        plt.show()