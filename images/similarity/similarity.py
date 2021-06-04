import os
from helper.utils import *
from models.res_similar import SimilarRes18
from PIL import Image
from tqdm import tqdm


def find_similarity(input_dir, output):

    # Obtain the feature of each image
    rescale_image(input_dir, output)
    img2vec = SimilarRes18()
    allVectors = {}

    # Convert the image to array
    print("Converting images to feature vectors:")
    for image in tqdm(os.listdir(output)):
        I = Image.open(os.path.join(output, image))
        vec = img2vec.getVec(I)
        allVectors[image] = vec
    I.close()

    # Create the similarity matrix

    similarityMatrix = get_similarity_matrix(allVectors)
    similarNames, similarValues = top_entries(5, similarityMatrix)

    # Append the image to the array list
    img = []
    for i in os.listdir(input_dir):
        img.append(str(i))
    
    # Sort all images
    img.sort()
    
    # Find the similarity and save
    for i in img:
        plot_similar_images(input_dir, i, 5, 1, similarNames, similarValues)
    
    
find_similarity('./selected/', './rescaled/')
