from pattern import Checker

def main():
    # create a Checker object with resolution and tile size
    checker = Checker(resolution=200, tile_size=25)

    # draw the checkerboard pattern
    checker.draw()
    
    # display the checkerboard pattern
    checker.show()

    print("Checkerboard pattern shape:", checker.output.shape)