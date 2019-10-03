import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = np.zeros((400, 400, 3), dtype="uint8")

# Creating circle
cv2.circle(img, (200, 200), 80, (255, 0, 0), 3)

cv2.imshow('im2.jpg', img)

# Allows us to see image
# untill closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()