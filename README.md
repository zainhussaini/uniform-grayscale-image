When you take a color image and convert it to grayscale, you lose some information but in general you can still tell what the picture is pretty well:

| RGB Image                                        | Grayscale Image                                     |
|--------------------------------------------------|-----------------------------------------------------|
| ![RGB image](imgs/image-rgb.png?raw=true "RGB Image") | ![Gray image](imgs/image-gray.png?raw=true "Gray Image") |

What if you take the RGB image and convert it so that the colors are similar but when converted to a gray cale image it's all the same color?

| Converted RGB Image                                                      | Grayscale Image                                         |
|--------------------------------------------------------------------------|---------------------------------------------------------|
| ![Converted RGB image](imgs/converted-rgb.png?raw=true "Converted RGB Image") | ![Gray image](imgs/converted-gray.png?raw=true "Gray Image") |

This is an algorithm to convert any RGB image into a similar RGB image that looks uniform when converted to grayscale.

# Background

I will be using OpenCV's grayscale conversion as the benchmark. It uses the NTSC formula for converting RGB values to grayscale:

$$ Y = 0.299 R + 0.587 G + 0.114 B $$

This is based on how the average person perceives the brightness of red, green, and blue light.

The RGB color space can be modeled as a 3D cube, where each of the axes corresponds to one of the color channels. Each side ranges from 0 to 255, where $(0,0,0)$ corresponds to black and $(255, 255, 255)$ corresponds to white.

![RGB color cube](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/RGB_color_solid_cube.png/1920px-RGB_color_solid_cube.png)

Each gray value corresponds to a plane that cuts through the RGB cube. The following video shows the plane as you vary the target grayscale value. The line pure gray line is shown as well, from $(0,0,0)$ to $(255, 255, 255)$.

https://github.com/user-attachments/assets/51312ecf-7053-47c0-99c7-64fbc5562365

Since we want to convert RGB values but keep the color the same, we can think about them in the HSV space instead, which is often mapped to a cylinder:

![HSV color cylinder](https://upload.wikimedia.org/wikipedia/commons/3/33/HSV_color_solid_cylinder_saturation_gray.png)

The idea of "same color" is best represented by preserving the hue, while allowing flexibility in the saturation and value.

# Algorithm

The target grayscale value is the average of the original image's grayscale values.

$$ Y = \begin{bmatrix} 0.299 & 0.587 & 0.114 \end{bmatrix}\begin{bmatrix} R_0 & R_1 & R_2 & \dots & R_n \\\ G_0 & G_1 & G_2 & \dots & G_n  \\\ B_0 & B_1 & B_2 & \dots & B_n  \end{bmatrix} \\ $$

$$ Y_{goal} = \bar{Y} $$

In the ideal case, for each pixel we can just scale the RGB values by some factor $s$ so that their converted grayscale value is the target.

$$ \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} = s \begin{bmatrix} R_0 \\\ G_0 \\\ B_0 \end{bmatrix} \qquad Y_{goal} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \end{bmatrix}\begin{bmatrix} R \\\ G \\\ B \end{bmatrix} $$

However this often results in RGB values that exceed the 255 limit. Truncating these values to 255 results in a change in hue, so in these cases a more sophisticated approach that preserves hue is necessary.

In order to maintain hue, the RGB point can be moved in the $(R_0, G_0, B_0)$ direction which changes value, or in the $(1, 1, 1)$ direction which changes saturation.

$$ \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} =a \begin{bmatrix} R_0 \\\ G_0 \\\ B_0 \end{bmatrix} + b \begin{bmatrix} 1 \\\ 1 \\\ 1 \end{bmatrix} $$

Since now there's freedom in two dimensions (due to $a$ and $b$), the hue preserving space is a plane, which intersects with the target grayscale plane. The following video shows the intersection as you vary the hue preserving plane.

https://github.com/user-attachments/assets/2361803b-bea7-4377-a29e-2424a3c68a5c

Since RGB can move along the two vectors $(R_0, G_0, B_0)$ and $(1, 1, 1)$, the only vector it can't move along is the cross product of them:

$$ (R_0, G_0, B_0) \times (1, 1, 1) = \begin{bmatrix} G_0 - B_0 & B_0 - R_0 & R_0 - G_0 \end{bmatrix} $$

Therefore the line of intersection is defined by:

$$ \begin{bmatrix} 0.299 & 0.587 & 0.114 \\\ G_0 - B_0 & B_0 - R_0 & R_0 - G_0 \end{bmatrix} \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} = \begin{bmatrix} Y_{goal} \\\ 0 \end{bmatrix}$$

Any movement along the line of intersection satisfies the target grayscale and preservation of hue. Since this is an underdetermined system, the solution includes a nullspace vector, and is the following:

$$ \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} = x_t + n x_n$$

$x_t$ can be any solution. One obvious one is $(Y_{goal}, Y_{goal}, Y_{goal})$. We should use the solution we found from just scaling the RGB values, and find the minimum $n$ needed to satisfy the constraints.

$x_n$ is the null space of the matrix. It is:

$$ \begin{bmatrix} -0.701 R_0 + 0.587 G_0 + 0.114 B_0 \\\ 0.299 R_0 - 0.413 G_0 + 0.114 B_0 \\\ 0.299 R_0 + 0.587 G_0 + -0.886 B_0 \end{bmatrix} $$

The constraints can be written as:

$$ \begin{bmatrix} 1 & 0 & 0 \\\ 0 & 1 & 0 \\\ 0 & 0 & 1 \\\ -1 & 0 & 0 \\\ 0 & -1 & 0 \\\ 0 & 0 & -1 \end{bmatrix} \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} \geq \begin{bmatrix} 0 \\\ 0 \\\ 0 \\\ -255 \\\ -255 \\\ -255 \end{bmatrix}$$

Reduce these matrices to only the rows that failed, resulting in

$$ C \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} \geq d $$

Solve for the required $n$ to make each inequality pass:

$$ n = (d - C x_t) / (C x_n) $$

Pick the maximum $n$. The solution is:

$$ \begin{bmatrix} R \\\ G \\\ B \end{bmatrix} = x_t + n x_n$$
