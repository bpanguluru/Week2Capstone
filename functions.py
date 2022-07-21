import skimage.io as io
import numpy as np

def load_image(path_to_image : str):

    """
    Description: Loads the image at the input file path and returns it

    Parameters:
    ----------------------------------------------------------------
    path_to_image: str
        The path to the image as a string
    ----------------------------------------------------------------
    Returns:
        Image: the image as a numpy array

    """

    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    
    return image



def cos_distance(desc1 : np.ndarray, desc2 : np.ndarray):
    """
    Description: Calculates the cosine distance bewteen two descriptors

    Parameters:
    ----------------------------------------------------------------
    desc1: 
        The path to the image as a string

    desc2: 
        The path to the image as a string
    ----------------------------------------------------------------
    Returns:
        cos_dist: the cosine distance between the two input vectors

    """

    dot_prod = np.dot(desc1, desc2)
    magnitude = np.linalg.norm(desc1)*np.linalg.norm(desc2)
    cos_dist = 1 - dot_prod / magnitude

    return cos_dist
