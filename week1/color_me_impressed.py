import cv2 

# load image
cat_image = cv2.imread('taiwan_cat.jpeg')

if cat_image is None:
    print("Error: image not found or unable to read.")

# display image
cv2.imshow("Displayed Cat", cat_image)

# wait for key press before closing window
cv2.waitKey(0)
cv2.destroyAllWindows()
