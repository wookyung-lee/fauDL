import matplotlib.pyplot as plt  # if not already globally imported in your pattern.py
from pattern import Checker, Circle, Spectrum  # assuming pattern.py is in the same directory

def main():
    # Checker example
    checker = Checker(resolution=250, tile_size=25)
    checker.draw()
    checker.show()

    # Circle example
    circle = Circle(resolution=250, radius=50, position=(50,50))
    circle.draw()
    circle.show()

    # Spectrum example
    spectrum = Spectrum(resolution=250)
    spectrum.draw()
    spectrum.show()

# script runs only when executed directly, not when imported as a module
if __name__ == "__main__":
    main()
