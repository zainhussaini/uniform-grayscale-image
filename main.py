#!/usr/bin/env python3

from PIL import Image
import numpy as np

L = 85
img = np.array(Image.open('image.png'), dtype=np.float32)

vals = []
for row in range(img.shape[0]):
    rgb_original = img[row, :, :].T
    A = 1/1000 * np.array([299, 587, 114]).reshape((1, 3))
    b = L * np.ones((1, rgb_original.shape[1])) - A @ rgb_original
    rgb_new = np.linalg.lstsq(A, b, rcond=None)[0]
    vals.extend(rgb_new)
    rgb_new = rgb_new + rgb_original

    img[row, :, :] = rgb_new.T


# img_temp = Image.fromarray(img, 'RGB')
# img_temp.show()

print(np.max(img))
print(np.min(img))
print(np.max(vals))
print(np.min(vals))


#         A = 1/1000 * np.array([299, 587, 114])
#         b = L - 1/1000 * np.array([299, 587, 114]) @ rgb_original
#
#         num_vals = rgb_original.shape[1]
#         A = np.ones((num_vals, 1)) @ A.reshape((1, 3))
#         b = b.reshape((num_vals, 1))
#
#         rgb_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#         print(rgb_new.shape)
#         print(rgb_original.shape)
#         rgb_new = rgb_new.reshape((3,)) + rgb_original
#
#         img[row, :, :] = rgb_new
#
#     print(f"{row}/{img.shape[0]} ({int(row/img.shape[0])}%)")
#
# img.save()
# print(type(img))
# img = img.convert('LA')
# print(type(img))
# img.save('greyscale.png')
