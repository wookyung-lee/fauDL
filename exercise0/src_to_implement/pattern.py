import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution: int, tile_size: int):
        """
        Initializes the Checkerboard object with the given resolution and tile size.

        Parameters:
            resolution: number of pixels in each dimension
            tile_size: number of pixels an individual tile has in each dimension

        The class is designed to generate a checkerboard pattern using the specified 
        resolution and tile size, and will store the pattern in the `output` variable 
        after calling the `draw()` method.
        """
    
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        """
        Creates a checkerboard pattern with the given resolution and tile size.

        The method generates a checkerboard pattern where the top-left corner is black 
        (represented by 1), and the tiles alternate between black and white. It then 
        expands the pattern to the specified resolution by repeating and resizing the 
        tiles accordingly.

        Returns:
            np.ndarray: A numpy array representing the full checkerboard pattern, 
                        where each tile is expanded to the specified pixel size.
        """

        # resolution must be evenly dividable by 2*tile_size
        if self.resolution % (2*self.tile_size) != 0:
            raise ValueError("The resolution must be divisible by 2*tile_size") 

        # create a 2x2 checkerboard pattern, but with top left corner being black, i.e. 1
        pattern = np.array([[0, 1], [1, 0]]) ### ???
        
        # tile it to fill the resolution
        reps = self.resolution // (2*self.tile_size) 
        checker_tiles = np.tile(pattern, (reps, reps)) # e.g. np.tile([0,1], 2) = [0,1,0,1]
        
        # expand each tile to actual pixel size using the kronecker product
        self.output = np.kron(checker_tiles, np.ones((self.tile_size, self.tile_size)))
        return self.output.copy()

    def show(self):
        """
        Displays the checkerboard pattern using plt.imshow.

        The pattern stored in 'output' is displayed as a grayscale image
        The axes are turned off to only display the pattern.
        """

        if self.output is None:
            raise ValueError("Draw the pattern first using .draw()")

        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.title('Checkerboard')
        plt.show()

####################################################################################
class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        """
        Initializes the Circle object with the given parameters.

        Parameters:
            resolution: 
            radius: radius the circle
            position: x-, y-coordinate of the circle center

        The class can be used to generate a circle pattern on an image of the given resolution.
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    
    def draw(self):
        """
        Generates a binary image of a circle within a square grid based on the specified resolution, center, and radius.

        The method creates a grid of coordinates and computes the squared distance from each pixel to the center of the circle.
        It then compares the squared distance to the squared radius to determine whether each pixel lies inside or outside the circle.

        The resulting binary image is stored in the `self.output` instance variable, where pixels inside the circle are set to 1 (True),
        and pixels outside the circle are set to 0 (False).

        Returns:
            numpy.ndarray: A copy of the generated binary circle image with shape (resolution, resolution), 
                        where 1 represents a pixel inside the circle, and 0 represents a pixel outside the circle.
        """
        
        # create a grid of coordinates
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)

        # stores x, y coordinates separately
        xx, yy = np.meshgrid(x, y)

        x_center = self.position[0]
        y_center = self.position[1]

        # compute squared distance from the center
        dist_sq = (xx - x_center)**2 + (yy - y_center)**2

        # each component of dist_sq will be compared with radius and
        # accordingly to the boolean value of the comparison, 
        # a binary image of the circle is generated
        self.output = (dist_sq <= self.radius**2).astype(np.uint8)

        return self.output.copy()

    def show(self):
        """
        Displays the checkerboard pattern using plt.imshow.

        The pattern stored in 'output' is displayed as a grayscale image
        The axes are turned off to only display the pattern.
        """

        if self.output is None:
            raise ValueError("Circle has not been drawn yet. Call draw() first.")

        plt.imshow(self.output, cmap='gray')
        plt.title("Binary Circle Image")
        plt.axis('off')
        plt.show()

####################################################################################
class Spectrum:
    def __init__(self, resolution: int):
        """

        Initializes a SpectrumImage object with the specified resolution.

        Parameter:
            resolution: The resolution (width and height) of the image. 
                           The generated image will have a size of (resolution x resolution).

        The class is designed to generate an RGB spectrum pattern based on the given 
        resolution, and the generated image will be stored in the `output` variable 
        after calling the `draw()` method.
        """
        self.resolution = resolution
        self.output = None
    
    def draw(self): 
        """
        Generates an RGB spectrum image based on the specified resolution.

        The method creates a smooth gradient for each of the RGB channels:
        - The Red channel (R) increases horizontally across the image.
        - The Green channel (G) increases vertically down the image.
        - The Blue channel (B) is determined by the absolute difference between the Red and Green gradients.

        The resulting RGB image is stored in the instance variable `self.output` and 
        a copy of the image is returned.

        Returns:
            numpy.ndarray: A copy of the generated RGB spectrum image with shape (resolution, resolution, 3),
                        where the third dimension represents the three color channels (R, G, B).
        """
        # generate a 2D grid of pixel coordinates
        x = np.linspace(0, 1, self.resolution)  # horizontal gradient (red)
        y = np.linspace(0, 1, self.resolution)  # vertical gradient (green)

        # create meshgrid for 2D coordinate space
        xx, yy = np.meshgrid(x, y)

        # initialize an empty RGB image
        image = np.zeros((self.resolution, self.resolution, 3))

        # assign the values to each channel based on xx and yy
        image[..., 0] = xx  # red channel (xx controls the intensity horizontally)
        image[..., 1] = yy # green channel (yy controls the intensity vertically)
        image[..., 2] = np.abs(xx - yy)  # blue channel (difference between xx and yy)

        self.output = image

        return self.output.copy() 

    def show(self):
        """
        Displays the checkerboard pattern using plt.imshow.

        The pattern stored in 'output' is displayed as a RBG image
        The axes are turned off to only display the pattern.
        """

        if self.output is None:
            raise ValueError("Circle has not been drawn yet. Call draw() first.")

        plt.imshow(self.output)
        plt.title("Specturm Image")
        plt.axis('off')
        plt.show()

# testing
test_object = Circle(1024, 200, (512, 256))  # Create a test object
test_object.draw()  # Generate the image
test_object.show()  # Display the generated image