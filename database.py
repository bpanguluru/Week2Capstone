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


from multiprocessing.sharedctypes import Value
import numpy as np
from camera import take_picture
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from facenet_models import FacenetModel
import pickle
import functions
from collections import defaultdict
model = FacenetModel()

database = defaultdict(None)
threshhold = 0.6 #change later
cutoff_probability = 0.95

def camera_to_array():
    """
    returns the numpy array of RGB values of the photo
    """

    print("SMILE!")
    imgarray = take_picture()
    return imgarray

def locate_faces(img_array):
    """
    pic is np array of shape (R, C, 3)
    (RGB is last dimension)
    """
    #need to clarify what R and C are
    boxes, probabilities, landmarks = model.detect(img_array)
    # input_des = model.compute_descriptors(img_array, boxes)
    


    cutoff_probs = probabilities >= 0.95

    return boxes[cutoff_probs], probabilities[cutoff_probs], landmarks[cutoff_probs]
    #return input_des, boxes, probabilities (probability cutoff: 95%)
    
    """
    for box, prob, landmark in zip(boxes, probabilities, landmarks):
    # draw the box on the screen
    ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))   
    """

def return_descriptors(image_array, boxes):
    input_descriptors = model.compute_descriptors(image_array, boxes)

    return input_descriptors

 #  determine if there is a match for each face in the database
def find_match(image_array):

    boxes, probabilities, landmarks = locate_faces(image_array)
    input_descriptors = model.compute_descriptors(image_array, boxes)

    matched_boxes = defaultdict(None)    
        
    for input_descriptor in input_descriptors:
        #calculate cosine distance between descriptor and average of each id's vectors 
        compare_dist = []

        for descriptor_list in database.values():
            cos_dist = functions.cos_distance(descriptor_list.mean(), input_descriptor)
            compare_dist.append(cos_dist)
            
        min_dist = min(compare_dist)
        index = compare_dist.index(min_dist)
        name = list(database.keys())

        if min_dist < threshhold:
            # may raise an error!!!!!!!!!!!!!!!!!!!
            matched_boxes[name[index]] = boxes[np.where(input_descriptors == input_descriptor)[0]]
            #ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))
        else:
            print("Person not found")
            new_name = input("Enter name for new person: ") 
            addProfile(new_name, input_descriptor)
            
            # Error Likely!!!!!!

            matched_boxes[new_name] = boxes[np.where(input_descriptors == input_descriptor)[0]]

    # Displaying the image with the boxes around the faces
    # ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))

    fig, ax = plt.subplots()
    ax.imshow(image_array)
    
    for box in boxes:
    # draw the box on the screen
        left, top, right, bottom = box
        width = right-left
        height = top-bottom
        ax.add_patch(Rectangle((left,bottom), width, height, fill=None, lw=2, color="red"))
        

    for key in matched_boxes:

        print("matched_boxes: ")
        print(matched_boxes[key])

        left, top, right, bottom = matched_boxes[key][0]
        ax.text(left, bottom, key, fontsize=8)

            
#functions to add or remove profiles
def addProfile(name: str, descriptors=None):
    #let user input name
    if descriptors is not None:
        if name in database:
            np.vstack((database[name], descriptors))
        else:
            database[name] = descriptors


def removeProfile(key: str):
    if key not in database.keys():
        print("you are trying to remove a profile that is not in the database >:(")
    else:   
        del(database[key])

def saveProfile():
    with open("database.pkl", mode="wb") as opened_file:
        pickle.dump(database, opened_file)

def loadProfile():
    with open("database.pkl", mode="rb") as opened_file:
        database1 = pickle.load(opened_file)
    return database1

class Profile:
    def __init__(self, name, descriptors=None):

        """This initializes the profile class with the name and descriptors provided.
        
        Parameters
        ----------
        name: Name associated with the profile

        descriptors
        """

        self.name = name
        if descriptors is None:
            self.descriptors = []
        else: 
            self.descriptors = descriptors

    def mean_descs(self):
        return self.descriptors.mean()