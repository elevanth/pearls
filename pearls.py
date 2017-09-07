# Every stuff for something tiny, forgetable but useful
# =============================================================

# If you wish to use CV2, you need to use the resize function.
# For example, this will resize both axes by half:

small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
# and this will resize the image to have 100 cols (width) and 50 rows (height):

resized_image = cv2.resize(image, (100, 50)) 
# Another option is to use scipy module, by using:

small = scipy.misc.imresize(image, 0.5)


