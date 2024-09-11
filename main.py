#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# constants
IMAGE_FILE = 'imgs/image-rgb.png'
GRAY_MATRIX = (0.299, 0.587, 0.114)

# open file
img = np.array(Image.open(IMAGE_FILE), dtype=np.float64)

# make matrix where each column is a pixel is each row is a channel (red, green, blue)
r = img[:, :, 0].flatten()
g = img[:, :, 1].flatten()
b = img[:, :, 2].flatten()
RGB0 = np.vstack((r, g, b))

# initialize output matrix
RGB1 = np.zeros(RGB0.shape)

# matrix to convert RGB values to gray
G = np.array(GRAY_MATRIX).reshape((1,3))
L = np.mean(G @ RGB0)

cache = dict()
def solve_with_cache(r0, g0, b0):
  cache_hash = r0 + 256*g0 + (256**2)*b0
  if cache_hash in cache:
    return cache[cache_hash]
  
  solution = solve(r0, g0, b0)
  cache[cache_hash] = solution
  return solution

def solve(r0, g0, b0):
  # handle degenerate case where color is purely a shade of gray
  if r0 == b0 == g0:
    return L*np.ones(3)

  # solve for scale factor that results in the right grayscale value
  s = L/(GRAY_MATRIX[0]*r0 + GRAY_MATRIX[1]*g0 + GRAY_MATRIX[2]*b0)

  # solve for initial solution, which might not satisfy inequality constraints.
  x_start = s*np.array([r0, g0, b0])

  # check if result is within bounds
  if np.all(x_start < 255*np.ones(3)):
    return x_start
  
  # find vector that points along line of intersection between grayscale and hue-preserving planes
  x_n = np.dot(GRAY_MATRIX, np.array([r0, g0, b0])) * np.ones(3) - np.array([r0, g0, b0])

  # point vector towards inside the bounds
  x_t = L*np.ones(3)
  if np.dot(x_start - x_t, x_n) > 0:
    x_n = -x_n

  # solve for n such that C @ (x_start + n*x_n) >= d
  C = np.vstack((np.eye(3), -np.eye(3)))
  d = np.array([0, 0, 0, -255, -255, -255])
  n = (d - C @ x_start)/(C @ x_n) * np.sign(C @ x_n)

  # find maximum value since this will satisfy all limits.
  n = np.max(n)

  # find optimal point
  x_fixed = x_start + n*x_n

  return x_fixed

# iterate over every column
for i in tqdm(range(RGB0.shape[1])):
  r0, g0, b0 = RGB0[:,i]
  RGB1[:,i] = solve_with_cache(r0, g0, b0)

# --------------------------------------------------------
# save the images
# --------------------------------------------------------

# round to nearest integer
RGB1 = (RGB1 + 0.5).astype(np.uint8)

r = RGB1[0, :].reshape(img.shape[:-1])
g = RGB1[1, :].reshape(img.shape[:-1])
b = RGB1[2, :].reshape(img.shape[:-1])
img = np.dstack((r, g, b))

img_gray = np.array(Image.fromarray(img).convert("L"))
print(f"resulting gray range: [{np.min(img_gray)}, {np.max(img_gray)}]")

Image.open('imgs/image-rgb.png').convert("L").save("imgs/image-gray.png")
Image.fromarray(img).save('imgs/converted-rgb.png')
Image.fromarray(img).convert("L").save('imgs/converted-gray.png')
