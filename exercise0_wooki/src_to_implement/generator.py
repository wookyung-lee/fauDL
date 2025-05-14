import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
import random


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: list, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # initialize epoch to track how many epoch we are at
        self.epoch = 0

        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        # Load file and labels (Assuming the labels are stored in a JSON file)
        self.image_files = sorted(os.listdir(file_path))
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # Get all image filenames and their corresponding labels
        self.image_filenames = list(self.labels.keys())
        self.num_images = len(self.image_filenames)
        
        # Shuffle indices if needed
        if self.shuffle:
            self.indices = list(range(self.num_images))
            random.shuffle(self.indices)
        else:
            self.indices = list(range(self.num_images))

        
        # # List all image files in the provided directory
        # self.image_files = [f for f in os.listdir(self.file_path) if f.endswith('.png') or f.endswith('.jpg')]

        # # Shuffle indices if needed
        # if self.shuffle:
        #     random.shuffle(self.image_files)

        # Initialize iterator
        self.current_index = 0


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        
        batch_images = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.current_index >= len(self.image_files):
                print("We have reached the end of the dataset, i.e. completed an epoch.")

                # reset the index to 0 so we can start from the beginning of the dataset
                self.current_index = 0 

                # to trach which epoch we are at
                self.epoch += 1

                # if self.shuffle:
                #     random.shuffle(self.indices)
                

            img_name = self.image_files[self.indices[self.current_index]]
            img_path = os.path.join(self.file_path, img_name)
            image = io.imread(img_path)

            # skimage.transform.resize (=! reshape)
            image = resize(image=image, output_shape=self.image_size) #, mode='reflect', anti_aliasing=True)

            
            label = self.labels[img_name]

            batch_images.append(image)
            batch_labels.append(label)
            self.current_index += 1

        ### 

        # count = 0
        # while count < self.batch_size:
        #     if self.index >= len(self.image_names):
        #         # End of epoch: reset
        #         self.index = 0
        #         if self.shuffle:
        #             random.shuffle(self.image_names)

        #     image_name = self.image_names[self.index]
        #     image_path = os.path.join(self.file_path, image_name)

        #     # Load and process image
        #     img = Image.open(image_path).convert('RGB')
        #     img = np.array(img)
            
        #     # Resize using interpolation
        #     img_resized = resize(
        #         img,
        #         output_shape=tuple(self.image_size),  # e.g., (64, 64, 3)
        #         preserve_range=True,
        #         anti_aliasing=True
        #     )
        #     img_resized = img_resized.astype(np.float32) / 255.0  # Normalize

        #     # Optional: augmentation
        #     if self.mirroring and random.choice([True, False]):
        #         img_resized = np.fliplr(img_resized)

        #     if self.rotation:
        #         angle = random.choice([0, 90, 180, 270])
        #         k = angle // 90
        #         img_resized = np.rot90(img_resized, k)

        #     # Add to batch
        #     batch_images.append(img_resized)
        #     batch_labels.append(self.labels[image_name])

        #     self.index += 1
        #     count += 1

        # return a tuple of (images, labels)
        return np.array(batch_images), np.array(batch_labels) # or ( np.array(batch_images), np.array(batch_labels) )?


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            img = np.fliplr(img)

        if self.rotation:
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, k=angle // 90)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, int_label):
        # This function returns the class name for a specific input

        return self.class_dict.get(int_label)       


    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        images, labels = self.next()  # Generate a batch

        batch_size = len(images)
        cols = min(5, batch_size)  # Display 5 images per row max
        rows = (batch_size + cols - 1) // cols  # Compute number of rows needed

        plt.figure(figsize=(15, 3 * rows))
        
        for i in range(batch_size):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')

        plt.tight_layout()
        plt.show()

