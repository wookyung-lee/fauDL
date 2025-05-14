import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.file_path = os.path.join('./data/', file_path)
     
        self.image_filenames = [filename for filename in os.listdir(self.file_path)]
        
        self.label_path = os.path.join('./data/', label_path)
        with open(self.label_path, 'r') as file:
            self.labels = json.load(file)

        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # if self.shuffle:
        #     random.shuffle(self.image_filenames)

        self.current_index = 0
        self.epoch_index = 0
        # self.end_epoch = True
        self.Num_batches = len(self.image_filenames) // self.batch_size
        


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        
        # if self.end_epoch:
        #     self.current_index = 0
        #     self.end_epoch = False
    
    # Shuffle the filenames at the beginning of each epoch if shuffle is enabled
        if self.current_index == 0 and self.shuffle:
            random.shuffle(self.image_filenames)
        
        images = []  # Batch of images
        labels = []  # Array with corresponding labels

        # Create a batch
        for _ in range(self.batch_size):
            # Check if we have reached the end of the dataset
            if self.current_index >= len(self.image_filenames):
                # Reset to the beginning and increment epoch index
                self.current_index = 0
                self.epoch_index += 1
                if self.shuffle:
                    random.shuffle(self.image_filenames)  # Shuffle again for the new epoch

            # Load the image and label if there are still images left
            file_name = self.image_filenames[self.current_index]
            src = np.load(f"{self.file_path}/{file_name}")
            images.append(self.augment(src))
            labels.append(self.labels[file_name.replace('.npy', '')])
            self.current_index += 1

        # Return the batch as numpy arrays
        return np.array(images), np.array(labels)


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if img.shape != self.image_size:
            img = skimage.transform.resize(img, self.image_size)
        if self.mirroring:
            mirroring_tyoe = random.choice(("lr", "ud"))
            if mirroring_tyoe == "lr":
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        if self.rotation:
            rotation_type = np.random.choice((90, 180, 270))
            if rotation_type == 90:
                img = np.rot90(img, 1)
            elif rotation_type == 180:
                img = np.rot90(img, 2)
            elif rotation_type == 270:
                img = np.rot90(img, 3)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_index

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.labels[str(x)]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        
        cols = 3
        rows = self.batch_size//3 + 1
        fig, ax = plt.subplots(rows, cols)
        axes = ax.flatten()           #2D array to one dimensional

        for i in range(self.batch_size):
            img = images[i]
            lab = self.class_dict[labels[i]]
            ax = axes[i]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab)

        # unused subplots
        for j in range(self.batch_size, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

# x = ImageGenerator("exercise_data", "Labels.json", 10, (32,32,3), True, True, True)
# x.show()