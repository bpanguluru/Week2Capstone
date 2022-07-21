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
from matplotlib.patches import Rectangle
from facenet_models import FacenetModel
import pickle
import functions
from collections import defaultdict
model = FacenetModel()

database = defaultdict(None)
threshhold = 1 #change later

class Profile:
    def __init__(self, name, descriptors=None): #

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

    def mean_descs():
        return self.descriptors.mean()
    
 

def camera_to_array():
    """
    returns the numpy array of RGB values of the photo
    """
    imgarray = take_picture()
    return imgarray

img_array=camera_to_array(pic)
def locate_faces(img_array):
    """
    pic is np array of shape (R, C, 3)
    (RGB is last dimension)
    """
    #need to clarify what R and C are
    boxes, probabilities, landmarks = model.detect(img_array)

    input_des = model.compute_descriptors(img_array, boxes)
    return input_des


 #  determine if there is a match for each face in the database
def find_match(input_descriptors):
    matched_descriptors = defaultdict(None)
    unmatched_descriptors = []
    for input_descriptor in input_descriptors:
        #calculate cosine distance between descriptor and average of each id's vectors 
        compare_dist = []
        for descriptor_list in database.values():
            cos_dist = functions.cos_distance(descriptor_list.mean(), input_descriptor)
            compare_dist.append(cos_dist)
        min_dist = min(compare_dist)
        index = compare_dist.index(min_dist)
        if min_dist < threshhold:
            matched_descriptors[index] = input_descriptor
        else:
            print("Person not found")
            new_name = input("Enter name for new person: ") 
            addProfile(new_name, input_descriptors)
    for key, descriptor in matched_descriptors:
        print(descriptor.shape)
        print(descriptor)
        #need to somehow print key with shape (key is the name of person)
    
            


#functions to add or remove profiles
def addProfile(name: string, descriptors=None):
    #let user input name
    if descriptors is not None:
        database[name] = descriptors
    else:
        database[name] = []  

def removeProfile(key: string, values: list):
    if key not in database.keys():
        print("you are trying to remove a profile that is not in the database >:(")
    else:   
        values = database[key]
        if len(values) == 1:
            del(database[key])
        else:
            values.remove(value)
            database[key] = values

def saveProfile():
    with open("database.pkl", mode="wb") as opened_file:
        pickle.dump(database, opened_file)

def loadProfile():
    with open("database.pkl", mode="rb") as opened_file:
        database1 = pickle.load(opened_file)
    return database1

        
#function to add descriptors?
# To-Do: OK
# can you guys do this in discord

# no >:(