import cv2
import numpy as np
import matplotlib.pyplot as plt

fname = 'C:/Users/Eleftherios/Downloads/dorin.png'
img = cv2.imread(fname)

luv_image = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

# # Display the LUV image
# cv2.imshow('LUV Image', luv_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

L = luv_image[..., 0]
U = luv_image[..., 1]
V = luv_image[..., 2]

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(L, U, V, c='r', marker='.', alpha=0.01)

# Set axis labels and title
ax.set_xlabel('L')
ax.set_ylabel('U')
ax.set_zlabel('V')
ax.set_title('3D Scatter Plot of LUV')

plt.show()



tmp = 1