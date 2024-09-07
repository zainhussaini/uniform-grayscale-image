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
  # handle degenerate case where color is purely gray
  if r0 == b0 == g0:
    return L*np.ones(3)

  # create system of equations to find initial solution
  # 1) 0.299 r + 0.587 g + 0.114 b = L
  # 2) r = r0*s
  # 3) g = g0*s
  # 4) b = b0*s

  A = np.array([
    [*GRAY_MATRIX, 0],
    [1, 0, 0, -r0],
    [0, 1, 0, -g0],
    [0, 0, 1, -b0],
  ])

  b = np.array([[L], [0], [0], [0]])

  # solution is [r; g; b; s] so extract first three
  x_start = (np.linalg.inv(A) @ b)[:3,:]

  # check inequality
  C = np.vstack([
    np.eye(3),
    -np.eye(3),
  ])

  d = np.vstack([
    np.zeros((3, 1)),
    -255*np.ones((3, 1)),
  ])

  # finish if solution satisfies inequality constraint
  if np.all(C @ x_start >= d):
    return x_start.flatten()

  # find specific inequality that wasn't satisfied
  inds = np.where(C @ x_start < d)[0]

  # reduce C and d to 1 dimensional matrices
  C_reduced = C[inds, :]
  d_reduced = d[inds, :]

  # find vector of line between two planes
  x_n = np.array([
    [((GRAY_MATRIX[0]-1)*r0 + GRAY_MATRIX[1]*g0 + GRAY_MATRIX[2]*b0)],
    [(GRAY_MATRIX[0]*r0 + (GRAY_MATRIX[1]-1)*g0 + GRAY_MATRIX[2]*b0)],
    [(GRAY_MATRIX[0]*r0 + GRAY_MATRIX[1]*g0 + (GRAY_MATRIX[2]-1)*b0)],
  ])

  n = (d_reduced - C_reduced @ x_start)/(C_reduced @ x_n)

  if np.any(n < 0):
    raise Exception(f"Negative n")
  
  n = np.max(n)
  x_fixed = x_start + n*x_n

  return x_fixed.flatten()

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
print("resulting gray range:", np.min(img_gray), np.max(img_gray))

Image.open('imgs/image-rgb.png').convert("L").save("imgs/image-gray.png")
Image.fromarray(img).save('imgs/converted-rgb.png')
Image.fromarray(img).convert("L").save('imgs/converted-gray.png')
