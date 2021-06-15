import os
from helper import utils as ut
from models.resnet18 import Resnet18
from PIL import Image
from tqdm import tqdm
import time


class Similar:

    def __init__(self, model: Resnet18 = None, input_dir='', output_dir='') -> None:
        self.model = Resnet18() if model is None else model
        self.input_dir = input_dir
        self.output_dir = output_dir

    def find_similarity(self, input_dir=None, output=None):

        # Assign the images input directory
        # and the output directory to store the rescaled images
        input_dir = self.input_dir if input_dir is None else input_dir
        output = self.output_dir if output is None else output

        # Obtain the feature of each image
        print("Scaling the image to obtain the feature ...")
        ut.rescale_image(input_dir, output)
        img2vec = self.model
        all_features = {}

        # Convert the image to array
        print("Converting images to feature vectors ...")
        for image in tqdm(os.listdir(output)):
            img = Image.open(os.path.join(output, image))
            feature = img2vec.get_feature_array(img)
            all_features[image] = feature
        img.close()

        # Create the similarity matrix
        print("Creating the similarity matrix ...")
        matrix = ut.get_similarity_matrix(all_features)
        label, score = ut.top_entries(4, matrix)

        # Append the image to the array list
        img = []
        for i in os.listdir(input_dir):
            img.append(str(i))

        # Sort all images
        # img.sort()

        # Find the similarity and save
        print("Saving images ...")
        beg = time.thread_time()
        for i in img:
            ut.plot_similar_images(input_dir, i, 4, 1, label, score)
        print(f"Completed in {time.thread_time() - beg} s", )


if __name__ == '__main__':
    Similar().find_similarity('../images/selected/', '../images/rescaled/')