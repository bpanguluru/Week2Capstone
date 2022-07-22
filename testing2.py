import database as db

img = db.camera_to_array() #should take photo
boxes, probabilities, landmarks = db.locate_faces(img)
db.find_match(img)