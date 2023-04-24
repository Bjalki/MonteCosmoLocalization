'''
Credit belongs to http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/#code
'''

#!/usr/bin/python
from string import hexdigits
from turtle import width
import cv2
import pandas as pd     
import random
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from PIL import Image
import sys
import math 
import statistics as stat
import kidnap
import img_processing as imgPr

# Arbitrary values, to model gaussian noise.
sensorVariance = 0.01
proportionalMotionVariance = 0.01

def MCL(robot: cozmo.robot.Robot):
  panoPixelArray = cv2.imread("cozmo-images-kidnap\c-Panorama.jpg") #image to read in, should read in our pano (the cropped one)
  panoPixelArray.astype("float")                                        #Make sure to change other references to desired image as needed in this file
  dimensions = panoPixelArray.shape
  width = dimensions[1]
  height = dimensions[0]
  # Initialize cozmo camera
  robot.camera.image_stream_enabled = True
  pixelWeights = [] # predictions

  # Algorithm MCL Line 2
  # fill array with uniform values as starting predictions at initialize
  particles = []  # starts as randomized particles, aka guesses for where the robot is facing
  M = 500   # Number of particles
  i = 0     # Iterations
  while i < M:
      particles.append(random.randint(0, 39))
      i = i + 1
  # Saves preliminary predictions to a dataframe
  pointFrame = pd.DataFrame(particles, columns=['particles'])
  
  i = 0
  while i < 10: # time steps is arbitrary
 
    robot.turn_in_place(degrees(9.0)).wait_for_completed() #collect 40 images so 9 degrees per turn
    cv_cozmo_image2 = None

    
    kidnap.takeSingleImage(robot) ##Uncomment me when done testing##################################
    cv_cozmo_image2 = imgPr.get_img("cozmo-images-kidnap\kidnapPhoto.jpg") #get kidnapPhoto from prev method
    

    # empty arrays that hold population number, and weight
    pixelPopulationNumber = []
    pixelWeights = []

    # empty that can hold new population, temp array list for the one above
    newParticles = []

    # Algorithm MCL Line 3
    for pose in particles:
      # Algorithm MCL Line 4
      newPose = sample_motion_model(pose, width)
      # Algorithm MCL line 5:
      # map is [0, 1] interval space for movement, sensing distance from 0
      weight = measurement_model(cv_cozmo_image2, newPose) 
      
      # Algorithm MCL line 6:
      pixelWeights = np.append(pixelWeights,[weight])
      pixelPopulationNumber = np.append(pixelPopulationNumber,[newPose])

    # Compute probabilities (proportional to weights) and cumulative distribution function for sampling of next pose population
    # NOTE: This is the heart of weighted resampling that is _not_ given in the text pseudocode.
    # - first sum weights

    # sum all weight, create new array size M, calculate probability
    sum_weights = 0.0
    for pixel in pixelWeights:
      sum_weights += pixel
    probabilities = []

    for m in pixelWeights:
      probabilities.append(m / sum_weights)

    #Cumulative Distribution Function
    cdf = []
    sum_prob = 0

    for prob in probabilities:
        sum_prob += prob
        cdf.append(sum_prob)
    # cdf[len(probabilities)] = 1.0 last is always 1.0

   # redistribute population to newX

   #Algorithm MCL line 8:
   #resampling
    for m in range(M):
        p = random.uniform(0, 1)
        index = 0
        while p >= cdf[index]:
            index += 1
        newParticles.append(pixelPopulationNumber[index])
    particles = newParticles
    i += 1
  newParticles.sort()

  # updating the CSV file with the original predictions and the newest predictions
  df = pd.DataFrame(newParticles, columns = ['newParticles'])
  df = df.join(pointFrame) # joins new predictions with original predictions
  df = df.sort_values(by=['newParticles'], ascending=False)
  df.to_csv("data/data.csv", index = False)
  
  print()


def sample_motion_model(xPixel, width):
  # making variance proportional to magnitude of motion command
  newX = xPixel + sample_normal_distribution(abs(proportionalMotionVariance))
  return max(0, min(width, newX))

# map in this case is [0, 1] allowable poses #Change particlePos to be index of curent img
def measurement_model(latestImage, particlePose):
    # Gaussian (i.e. normal) error, see https://en.wikipedia.org/wiki/Normal_distribution
    # same as p_hit in Figure 6.2(a), but without bounds. Table 5.2
  #img = Image.open("cozmo-images-kidnap\c-Panorama.jpg")
  #width, height = img.size
  #get the slice of the panorama that corresponds to the pixel
  particle = Image.open("cozmo-images-kidnap" + particlePose) #slice(img, particlePose, 320, 320, height)
  particle = np.array(particle)
  #resize the images
  #cv_particle = cv2.resize(particle, (width, height))
  image2 = latestImage
  #compare how similar/different they are using MSE
  diff = compare_images(particle, image2)
  #see Text Table 5.2, implementation of probability normal distribution
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def sample_sensor_model(pose):
  # ideal sensor will return pose (distance to right of 0), but we'll model Gaussian noise (could have more components as in class)
  return pose + sample_normal_distribution(sensorVariance)

#see Text Table 5.4, implementation of sample normal distribution
def sample_normal_distribution(variance):
  sum = 0
  for i in range(12):
    sum += (2.0 * random.random()) - 1.0
  return math.sqrt(variance) * sum / 2.0

# Compares images by pixels using Mean Squared Error formula
def compare_images(imageA, imageB):
  # See https://en.wikipedia.org/wiki/Mean_squared_error 
  dimensions = imageA.astype("float").shape
  width = dimensions[1]
  height = dimensions[0]
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  # Dividing the values so they fit 
  err /= (width * height * width * height)
  return err


    
    
    
#CURRENTLY UNINCORPORATED###########################
#Updates our data for when we move the cozmo to take a new picture to compare to the panorama
def motionUpdate(): 
  #reads in the panorama and the data
  data = pd.read_csv("data/data.csv")
  pano = cv2.imread("cozmo-images-kidnap\c-Panorama.jpg")
  dimensions = pano.shape
  width = dimensions[1]
  
  #gives a rough estimate of how many pixels to the right we are moving in the panorama image
  toAdd = math.floor(width / 360)
  data['X'] = data['X'].apply(lambda x: x + (5 * toAdd))
  #data['X'] = data['X'] + (10 * toAdd)
  data.loc[data.X > (width - 320), 'X'] = random.randint(1, width - 330)  
  data.to_csv("data/data.csv", index = False)


# TO-DO



# if __name__ == '__main__':
#   # robot = cozmo.robot.Robot
#   # MCL(robot)