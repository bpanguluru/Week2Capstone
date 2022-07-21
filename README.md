# Week2Capstone
BWSI 2022 Week 2 Vision Capstone

database.py
  - Take input from camera
  - locate faces in the image and extract their “descriptor vectors”
  - determine if there is a match for each face in the database
  - return the image with rectangles around the faces along with the corresponding name (or “Unknown” if there is no match)
  - Profile class
    - functions to add/remove profiles
    - Name, list of descriptor vectors
  - Dictionary to store name/profile pairs
  - determine cosine distance threshhold

clustering.py
  - Create node class
    - descriptor vector, file path, unique id, label, list of neighbor Ids
  - Adjacency matrix
  - whipsers algorithm function
    - calculate cosine distances between pairs of nodes
    - Create edge between nodes if distance is under calculated threshhold 
    - randomly select and edge and update its label based off of its neighbors
      - repeat a certain number of times
    - return groups???
  - function that inputs stack of images and uses above classes/functions to return sorted images

functions.py
  - function to calculate cosine distance 
  - functions to load images as numpy arrays
