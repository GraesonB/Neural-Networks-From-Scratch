import numpy as np
import random
from PIL import Image
from matplotlib.image import imread
import os
from mlxtend.data import loadlocal_mnist

image_size = (128,128)
cat_path = 'PetImages/cats_resized'
cat_label = 0
dog_path = 'PetImages/dogs_resized'
dog_label = 1
npy_dir = 'PetImages/npy files'


# resizes a folder of images (jpg) to a uniform size
# inputs: directory of original set, target directory of resized set, returns nothing
def resize_set(directory):
    counter = 0
    resized_dir = directory + "/resized"
    if not os.path.isdir(resized_dir):
        os.mkdir(resized_dir)
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            try:
                img = Image.open(directory + "/" + file).convert("RGB")
                img_resize = img.resize(image_size)
                new_name = resized_dir + "/" + str(counter) + '.jpg'
                img_resize.save(new_name)
                counter +=1
            except:
                print( "[Pillow Error] Problem with: " + file + ". (maybe an empty image?)")

# turns images into np arrays
# inputs: a folder directory of images, and the desired label
def image_set_to_arrays(directory, label):
    X, Y = [], []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            array = imread(directory + "/" + file)
            X.append(array)
            Y.append(label)
    data = np.array([X, Y])
    return data

# combines a list of datasets created by images_to_arrays function and randomly orders the data
def combine_and_shuffle(list_of_datasets):
    combined_X, combined_Y = [], []
    shuffled_X, shuffled_Y = [], []
    for dataset in list_of_datasets:
        combined_X.extend(dataset[0])
        combined_Y.extend(dataset[1])
    random_idx_list = list(range(len(combined_X)))
    random.shuffle(random_idx_list)
    for random_idx in random_idx_list:
        shuffled_X.append(combined_X[random_idx])
        shuffled_Y.append(combined_Y[random_idx])
    complete_data = np.array([shuffled_X, shuffled_Y])
    return complete_data

def split_data(data, train_proportion = 0.6):
    X, Y = data[0], data[1]
    m = len(X)
    train_cutoff = int(np.floor(m * train_proportion))
    dev_cutoff = int(np.floor((m - train_cutoff) / 2)) + train_cutoff
    train = [np.stack(X[0:train_cutoff]), np.stack(Y[0:train_cutoff])]
    dev = [np.stack(X[train_cutoff:dev_cutoff]), np.stack(Y[train_cutoff:dev_cutoff])]
    test = [np.stack(X[dev_cutoff:]), np.stack(Y[dev_cutoff:])]
    return train, dev, test


def load_cats_dogs_25k():
    data = np.load(npy_dir + "/cats_dogs_25k.npy", allow_pickle = True)
    train, dev, test = split_data(data)
    return train, dev[0:10], test[0:10]

def load_cats_dogs_64():
    data = np.load(npy_dir + "/cats_dogs_25k.npy", allow_pickle = True)
    return data[0][:64], data[1][:64]

def load_cats_dogs_1k():
    data = np.load(npy_dir + "/cats_dogs_25k.npy", allow_pickle = True)
    return data[0][:2000], data[1][:2000]

if __name__ == "__main__":
    # train, dev, test = load_cats_dogs_25k()
    # X, Y = load_cats_dogs_64()
    # print("Train shape: " + str(np.shape(train[0])))
    # print('')
    #
    # print("Train set length: " + str(len(train[0])))
    # print("Dev set length: " + str(len(dev[0])))
    # print("Test set length: " + str(len(test[0])))
    # print('')
    train_X, train_Y = loadlocal_mnist(images_path='train-images-idx3-ubyte', labels_path='train-labels-idx1-ubyte')
    train_X = np.reshape(train_X, (train_X.shape[0], 28, 28))

    print(train_X.shape)
    print(train_Y.shape)
    print(train_X[0])
    # data, _, _ = load_cats_dogs_25k()
    # X = data[0]
    # Y = data[1]
    pic = 1011
    image = Image.fromarray(train_X[pic])
    print(train_Y[pic])
    image.show()
