"""
database.py(Jayashabari, Josh, Jamie)

Take input from camera - Jamie
locate faces in the image and extract their “descriptor vectors” - Jamie
determine if there is a match for each face in the database - Jayashabari

return the image with rectangles around the faces along with the corresponding name (or “Unknown” if there is no match) - Jayashabari
Profile class
functions to add/remove profiles
Name, list of descriptor vectors
Dictionary to store name/profile pairs
determine cosine distance threshhold
"""



from camera import take_picture
import matplotlib.pyplot as plt
from facenet_models import FacenetModel
import pickle

model = FacenetModel()

database = defaultdict(None)
threshhold = 1 #change later

def camera_to_array():
    """
    returns the numpy array of RGB values of the photo
    """
    img_array = take_picture()
    return img_array

def locate_faces(img_array):
    """
    pic is np array of shape (R, C, 3)
    (RGB is last dimension)
    """
    #need to clarify what R and C are
    boxes, probabilities, landmarks = model.detect(img_array)

    input_descriptors = model.compute_descriptors(img_array, boxes)
    return input_descriptors


 #   determine if there is a match for each face in the database
 for input_descriptor in input_descriptors:
    #calculate cosine distance between descriptor and average of each id's vectors 
    compare_dist = []
    for descriptor_list in database.values():
        cos_dist = cos_distance(descriptor_list.mean(), input_descriptor)
        compare_dist.append(cos_dist)
    min_dist = compare_dist.min()
    index = compare_dist.index(min_dist)
    if min_dist < threshhold:
        print(database.key)
    else:
        print("Person not found") 
            


