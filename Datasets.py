#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

#for error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[43]:


def get_model_inputs(img_objects, im_size, num_class):
    """
    Generates inputs for the model using image objects.
    Params:
        img_objects (list): List of image objects to feed the model.
        im_size (tuple): Image size used in the model. Width, heigth and channel information.
        num_class (int): number class in total.
    Returns:
        Sketch and real image arrays, class labels of images.
    """
    n_examples = len(img_objects)
    w,h,c = im_size
    sketches = np.empty((n_examples, w, h, c))
    real_imgs = np.empty((n_examples, w, h, c))
    class_labels = np.empty(n_examples)
    for i in range(n_examples):
        sketches[i], real_imgs[i] = img_objects[i].read_image( (w,h) )
        class_labels[i] = img_objects[i].class_label
    return sketches, real_imgs, to_categorical(class_labels, num_classes=num_class)


# In[44]:


def load_combined_image(im_path, t_size):
    """
    Load combined image. Crops image into half vertically then apply resize operation.
    Images will be returned with 4 dims as Keras model expect.
    Parameters:
        im_path (string): Path of the combined image
        t_size (tuple): New size of the images
    Returns:
        numpy array of the read image.
    """
    im = Image.open(im_path)
    width, height = im.size   # Get dimensions
    
    left = 0
    top = 0
    right = width/2
    bottom = height
    sketch_img = im.crop((left, top, right, bottom)).resize(t_size)
    
    left = width/2
    right = width
    real_img = im.crop((left, top, right, bottom)).resize(t_size)
    
    sketch_arr = img_to_array(sketch_img)
    real_arr = img_to_array(real_img)
    
    return np.expand_dims(sketch_arr, axis=0), np.expand_dims(real_arr, axis=0)


# In[51]:


def load_single_image(im_path, t_size):
    """
    Loads single image given using Keras load_img function.
    Image will be returned with 4 dims as Keras model expect.

    Params:
        im_path (string): Absolute path of the image to be loaded.
        t_size (tuple): Desired size of the image.
    Returns:
        numpy array of the read image.
    """
    img = load_img(im_path ,target_size=t_size)
    img = img_to_array(img)
    return np.expand_dims(img, axis=0)


# In[46]:


class ImageObj:
    def __init__(self, name, full_path, d_main_dir, is_combined, class_label):    
        """
        Creates an image object.
        Params: 
            full_path (str): Full path of the image file
            is_combined (bool): Whether images and its real versions included in same image or not
            class_num (int): Label of the class. Used in auxilary classification loss
        """
        self.name = name
        self.full_path = full_path
        self.dataset_main_dir =  d_main_dir
        self.is_combined = is_combined
        self.class_label = class_label
    
    def read_image(self, target_size):
        """
        Read image and its real one together
        Returns:
            sketch and real image as numpy arrays
        """
        real_img_dir = self.full_path
        if not self.is_combined:
            real_img_dir = os.path.join(self.dataset_main_dir, 'data', self.name)
            skecth_img = load_single_image(self.full_path, target_size)
            real_img = load_single_image(real_img_dir, target_size)
            return skecth_img, real_img
        return load_combined_image(self.full_path, target_size)


# In[47]:


class Dataset:
    def __init__(self, d_name, main_path, is_combined, class_num):
        """
        Creates a dataset object.
        Params: 
            main_path (str): Parent directory of the dataset
            is_combined (bool): Whether images and its real versions included in same image or not
            class_num (int): Label of the class. Used in auxilary classification loss
        """
        self.dataset_name = d_name
        self.main_parent_path = main_path
        self.is_combined = is_combined
        self.class_num = class_num
    
    def get_all_image_objects(self):
        # Construct parent directory
        parent_dir = self.main_parent_path
        if not self.is_combined:
            parent_dir = os.path.join(self.main_parent_path, 'edges_final')
        # Get name of the images
        image_names = os.listdir(parent_dir)
        
        img_objs = np.empty(len(image_names), dtype=object)
        # Generate image objects
        for i, im_name in enumerate(image_names):
            im_path = os.path.join(parent_dir, im_name)
            img_objs[i] = ImageObj(im_name, im_path, self.main_parent_path, self.is_combined, self.class_num)
        return img_objs
    
    

