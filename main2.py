#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = np.array(Image.open('image-rgb.png'), dtype=np.float64)
r = img[:, :, 0].flatten()
g = img[:, :, 1].flatten()
b = img[:, :, 2].flatten()

A = 1/1000 * np.array([299, 587, 114]).reshape((1, 3))
RGB0 = np.vstack((r, g, b))
# L = np.min(A @ np.eye(3)*255)
L = np.mean(A @ RGB0)
# L = 226
Ls = L*np.ones((1, RGB0.shape[1])) - A @ RGB0

# solve minimize ||RGB1 - RGB0||_2
# constraint 1: A * RGB1 = Ls
# constraint 2: 0 <= RGB1 <= 255

# define x = RGB1 - RGB0  =>  RGB1 = x + RGB0
# solve minimize ||x||_2
# constraint 1: A*x = Ls - A*RGB0
# constraint 2: 0 - RGB0 <= x <= 255 - RGB0

""" the following doesn't enforce constraint 2 """
x = np.linalg.lstsq(A, Ls, rcond=None)[0]
RGB1 = x + RGB0
RGB1_original = RGB1.copy()

""" the following needs 500+ TB """
# import cvxpy as cp
# import numpy as np
#
# # Problem data.
# N = RGB0.shape[1]
#
# # Construct the problem.
# RGB1 = cp.Variable((3, N))
#
# objective = cp.Minimize(cp.sum_squares(RGB1 - RGB0))
# constraints = [A*RGB1 - Ls == 0, 0 <= RGB1, RGB1 <= 255]
# prob = cp.Problem(objective, constraints)
#
# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(RGB1.value)
# # The optimal Lagrange multiplier for a constraint is stored in
# # `constraint.dual_value`.
# print(constraints[0].dual_value)

""" null space method """
from scipy.linalg import null_space

# there is a plane, defined as where A * RGB = L
# the points have been projected onto this plane, but they are not necessarily
# in the positive octant
# A0 is the unit vector perpendicular to plane
# A1 and A2 are orthogonal and parallel to plane
ns = null_space(A)
A0 = A.flatten()/np.linalg.norm(A)
A1 = ns[:, 0]
A2 = ns[:, 1]
# rotates from normal RGB axes to equal greyscale plane
Rotm = np.vstack((A0, A1, A2))

# smallest vector from origin to plane
middle = np.linalg.lstsq(A, L*np.ones((1,1)), rcond=None)[0]
# pixels in RGB1 where there are any negative values (these pixels need to be fixed)
inds = np.logical_or(np.logical_or(RGB1[0, :] < 0, RGB1[1, :] < 0), RGB1[2, :] < 0)
# number of pixels that need to be fixed
print(f"{np.sum(inds)}/{len(inds)} ({100*np.sum(inds)/len(inds)}%)")

bad_vals = RGB1[:, inds]
# project bad_vals onto 2D plane
bad_vals_yz_plane = Rotm @ (bad_vals - middle)
assert np.isclose(np.min(bad_vals_yz_plane[0, :]), 0)
assert np.isclose(np.max(bad_vals_yz_plane[0, :]), 0)

# find the vertices of the triangle that is defined by grayscale plane and positive octant
lims = np.diag((np.ones((1, 3))*L / A).flatten())
assert np.allclose(A @ lims, np.ones((1, 3))*L)

# project these points onto plane
verts = (Rotm @ (lims - middle))
assert np.isclose(np.min(verts[0, :]), 0)
assert np.isclose(np.max(verts[0, :]), 0)

# for points that are outside triangle, project to the triangle
for i in range(3):
    v1 = verts[1:, i]
    v2 = verts[1:, (i+1)%3]

    # 2D cross product matrix
    a = (v2 - v1) @ np.array([[0, 1], [-1, 0]])
    b = bad_vals_yz_plane[1:, :] - v1.reshape((2, 1))

    # determine indeces of all points outside this edge of triangle
    inds_outside_edge = (a @ b) < 0

    # calculate new points
    v12_norm = (v2 - v1)/np.linalg.norm(v2 - v1)
    projected = v12_norm.reshape((2, 1)) @ v12_norm.reshape((1, 2)) @ b + v1.reshape((2, 1))
    bad_vals_yz_plane[1:, inds_outside_edge] = projected[:, inds_outside_edge]

# convert back into RGB axes and set
RGB1[:, inds] = Rotm.T @ bad_vals_yz_plane + middle
assert np.allclose(A @ RGB1, L)
assert (np.isclose(np.min(RGB1), 0) or np.min(RGB1) >= 0)
assert np.max(RGB1) <= 255

# round to nearest integer
RGB1 = (RGB1 + 0.5).astype(np.uint8)

""" the following plots the histograms of images """
# colors = ["red", "green", "blue"]
# datas = [RGB0, RGB1_original, RGB1]
# 
# fig, axs = plt.subplots(nrows=len(colors), ncols=len(datas), sharex=True, sharey=True)
# for i, color in enumerate(colors):
#     # axs[i].hist(x[i, :], color=colors[i], bins="auto")
#     for j, data in enumerate(datas):
#         axs[i, j].hist(data[i, :], color=color, bins=256, range=(0,256))
#
# print("RGB0:R", np.min(RGB0[0,:]), np.max(RGB0[0,:]))
# print("RGB0:G", np.min(RGB0[1,:]), np.max(RGB0[1,:]))
# print("RGB0:B", np.min(RGB0[2,:]), np.max(RGB0[2,:]))
#
# print("RGB1:R", np.min(RGB1[0,:]), np.max(RGB1[0,:]))
# print("RGB1:G", np.min(RGB1[1,:]), np.max(RGB1[1,:]))
# print("RGB1:B", np.min(RGB1[2,:]), np.max(RGB1[2,:]))
# plt.show()
# exit()

r = RGB1[0, :].reshape(img.shape[:-1])
g = RGB1[1, :].reshape(img.shape[:-1])
b = RGB1[2, :].reshape(img.shape[:-1])
img = np.dstack((r, g, b))

img_gray = np.array(Image.fromarray(img).convert("L"))
print(np.min(img_gray))
print(np.max(img_gray))

Image.open('image-rgb.png').convert("L").save("image-gray.png")
Image.fromarray(img).save('converted-rgb.png')
Image.fromarray(img).convert("L").save('converted-gray.png')
