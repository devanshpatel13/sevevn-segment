# import cv2
# import pytesseract
# # import pdb
#
# # pdb.set_trace()
# pytesseract.pytesseract.tesseract_cmd = '/bin/tesseract'
# img = cv2.imread("/home/rajkplutus/PycharmProjects/text-detector/red-font.png")
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# abc = pytesseract.image_to_string(img)
# # print(int(abc))
# # print(len(abc))
# file = open("green.txt", "w+")
# file.write(abc)
# file.close()
# cv2.imshow('result', img)
# cv2.waitKey(0)

###########################################################
# import cv2
# import pytesseract
# import pdb
# pdb.set_trace()
# pytesseract.pytesseract.tesseract_cmd = r"/bin/tesseract"
#
# # Load image, grayscale, Otsu's threshold, then invert
# image = cv2.imread('green-font.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# invert = 255 - thresh
#
# # Perfrom OCR with Pytesseract
# data = pytesseract.image_to_string(invert)
# print(data)
#
# # cv2.imshow('thresh', thresh)
# cv2.imshow('invert', invert)
#
# # file = open("green1.txt", "w+")
# # file.write(data)
# # file.close()
# cv2.waitKey()


##############################
# import pytesseract
# import cv2
# img = cv2.imread('/home/rajkplutus/PycharmProjects/text-detector/abc.png')

# img = cv2.resize(img, (600, 360))
# print(pytesseract.image_to_string(img))
# abc = pytesseract.image_to_string(img)
# file = open("new.txt", "w+")
# file.write(abc)
# file.close()
# cv2.imshow('Result', img)
# cv2.waitKey(0)




# import cv2
# import numpy as np
#
# img = cv2.imread('Rectangle.png')
# # img = cv2.imread('SaleDigit.jpg')/
#
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for cnt in cnts:
#     approx = cv2.contourArea(cnt)
#     print(approx)
#
# cv2.imshow('image', img)
# cv2.imshow('Binary',thresh_img)
# cv2.waitKey()


import cv2
import numpy as np
#
# img = cv2.imread('SaleDigit.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret,thresh = cv2.threshold(gray,50,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# print("Number of contours detected:", len(contours))
#
# for cnt in contours:
#    x1,y1 = cnt[0][0]
#    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#    if len(approx) == 4:
#       x, y, w, h = cv2.boundingRect(cnt)
#       ratio = float(w)/h
#       if ratio >= 0.9 and ratio <= 1.1:
#          img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
#          cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#       else:
#          cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#          img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
#
# cv2.imshow("Shapes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
#
# img = cv2.imread('SaleDigit.jpg')
# imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
# contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#
#
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
#     cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1] - 5
#     if len(approx) == 3:
#         cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
#     elif len(approx) == 4 :
#         x, y , w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w)/h
#         print(aspectRatio)
#         if aspectRatio >= 0.95 and aspectRatio < 1.05:
#             cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#         else:
#             cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#     elif len(approx) == 5 :
#         cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 10 :
#         cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     else:
#         cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('SaleDigit.jpg',0)
# edges = cv.Canny(img,100,200)
# import pdb; pdb.set_trace()
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


# import sys
# import numpy
# from PIL import Image, ImageOps, ImageDraw
# from scipy.ndimage import morphology, label,grey_dilation
#
# def boxes(orig):
#     img = ImageOps.grayscale(orig)
#     im = numpy.array(img)
#
#     # Inner morphological gradient.
#     im = morphology.grey_dilation(im, (3, 3)) - im
#
#     # Binarize.
#     mean, std = im.mean(), im.std()
#     t = mean + std
#     im[im < t] = 0
#     im[im >= t] = 1
#
#     # Connected components.
#     lbl, numcc = label(im)
#     # Size threshold.
#     min_size = 200 # pixels
#     box = []
#     for i in range(1, numcc + 1):
#         py, px = numpy.nonzero(lbl == i)
#         if len(py) < min_size:
#             im[lbl == i] = 0
#             continue
#
#         xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
#         # Four corners and centroid.
#         box.append([
#             [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
#             (numpy.mean(px), numpy.mean(py))])
#
#     return im.astype(numpy.uint8) * 255, box
#
#
# orig = Image.open("SaleDigit.jpg")
# im, box = boxes(orig)
#
# # Boxes found.
# Image.fromarray(im).save("boxes.jpg")
#
# # Draw perfect rectangles and the component centroid.
# img = Image.fromarray(im)
# visual = img.convert('RGB')
# draw = ImageDraw.Draw(visual)
# for b, centroid in box:
#     draw.line(b + [b[0]], fill='yellow')
#     cx, cy = centroid
#     draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='red')
# visual.save("boxes1.jpg")


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# import os
#
# mypath = 'SaleDigit.jpg'
# onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
# images = np.empty(len(onlyfiles), dtype=object)
# for n in range(0, len(onlyfiles)):
#     images[n] = cv2.imread(os.path.join(mypath, onlyfiles[n]))
#
#     gwash = images[n]  # import image
#
#     gwashBW = cv2.cvtColor(gwash, cv2.COLOR_RGB2GRAY)  # change to grayscale
#
#     height = np.size(gwash, 0)
#     width = np.size(gwash, 1)
#
#     ret, thresh1 = cv2.threshold(gwashBW, 41, 255, cv2.THRESH_BINARY)
#
#     kernel = np.ones((1, 1), np.uint8)
#
#     erosion = cv2.erode(thresh1, kernel, iterations=31)
#     opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
#     _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     areas = []  # list to hold all areas
#
# for i, contour in enumerate(contours):
#     ar = cv2.contourArea(contour)
#     areas.append(ar)
#     cnt = contour
#     (x, y, w, h) = cv2.boundingRect(cnt)
# if cv2.contourArea(cnt) > 60000 and cv2.contourArea(cnt) < (height * width):
#     if hierarchy[0, i, 3] == -1:
#         cv2.rectangle(gwash, (x, y), (x + w, y + h), (255, 0, 0), 12)
#
# plt.subplot2grid((2, 5), (0, n)), plt.imshow(gwash)
# plt.title('Extraction'), plt.xticks([]), plt.yticks([])
#
# plt.show()





# import cv2
# import numpy as np
#
# img = cv2.imread('SaleDigit.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret,thresh = cv2.threshold(gray,50,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# print("Number of contours detected:", len(contours))
#
# for cnt in contours:
#    x1,y1 = cnt[0][0]
#    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#    if len(approx) == 15:
#       x, y, w, h = cv2.boundingRect(cnt)
#       ratio = float(w)/h
#       if ratio >= 0.9 and ratio <= 1.1:
#          img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
#          cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#       else:
#          cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#          img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
#
# cv2.imshow("Shapes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#
# import cv2
# def shapes():
#     img = cv2.imread('scale_machine.jpg')
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     white = np.ones((img.shape[0], img.shape[1], 3))
#
#     for c in contours:
#         approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
#         cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)
#         x = approx.ravel()[0]
#         y = approx.ravel()[1] - 5
#         if len(approx) == 3:
#             cv2.putText(img, "Triangle", (x, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#         elif len(approx) == 4:
#             x1, y1, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / float(h)
#             print(aspect_ratio)
#             if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
#                 cv2.putText(img, "Square", (x, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#             else:
#                 cv2.imshow("img", img)
#                 # import pdb; pdb.set_trace()
#                 cv2.putText(img, "Rectangle", (x, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#         elif len(approx) == 5:
#             cv2.putText(img, "Pentagon", (x, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#         elif len(approx) == 10:
#             cv2.putText(img, "Star", (x, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#         else:
#             cv2.putText(img, "Circle", (x, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#
#     cv2.imshow("Shapes", img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# shapes()


# import cv2
# import numpy as np
# font = cv2.FONT_HERSHEY_COMPLEX
#
# img = cv2.imread("scale_machine.jpg", cv2.IMREAD_GRAYSCALE)
# _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# _, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#     cv2.drawContours(img, [approx], 0, (0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 3:
#         cv2.putText(img, "Triangle", (x, y), font, 1, (0))
#     elif len(approx) == 4:
#         cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
#     elif len(approx) == 5:
#         cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
#     elif 6 < len(approx) < 15:
#         cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
#     else:
#         cv2.putText(img, "Circle", (x, y), font, 1, (0))
#
# cv2.imshow("shapes", img)
# cv2.imshow("Threshold", threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2  # OpenCV Library

# Image to detect shapes on below
image = cv2.imread("SaleDigit.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray image

# Setting threshold value to get new image (In simpler terms: this function checks every pixel, and depending on how
# dark the pixel is, the threshold value will convert the pixel to either black or white (0 or 1)).
_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

# Retrieving outer-edge coordinates in the new threshold image
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterating through each contour to retrieve coordinates of each shape
for i, contour in enumerate(contours):
    if i == 0:
        continue

    # The 2 lines below this comment will approximate the shape we want. The reason being that in certain cases the
    # shape we want might have flaws or might be imperfect, and so, for example, if we have a rectangle with a
    # small piece missing, the program will still count it as a rectangle. The epsilon value will specify the
    # precision in which we approximate our shape.
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Drawing the outer-edges onto the image
    cv2.drawContours(image, contour, 0, (0, 0, 0), 4)

    # Retrieving coordinates of the contour so that we can put text over the shape.
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + (w / 3))  # This is an estimation of where the middle of the shape is in terms of the x-axis.
    y_mid = int(y + (h / 1.5))  # This is an estimation of where the middle of the shape is in terms of the y-axis.

    # Setting some variables which will be used to display text on the final image
    coords = (x_mid, y_mid)
    colour = (0, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX

    # This is the part where we actually guess which shape we have detected. The program will look at the amount of edges
    # the contour/shape has, and then based on that result the program will guess the shape (for example, if it has 3 edges
    # then the chances that the shape is a triangle are very good.)
    #
    # You can add more shapes if you want by checking more lenghts, but for the simplicity of this tutorial program I
    # have decided to only detect 5 shapes.
    if len(approx) == 3:
        cv2.putText(image, "Triangle", coords, font, 1, colour, 1)  # Text on the image
    elif len(approx) == 4:
        cv2.putText(image, "Quadrilateral", coords, font, 1, colour, 1)
    elif len(approx) == 5:
        cv2.putText(image, "Pentagon", coords, font, 1, colour, 1)
    elif len(approx) == 6:
        cv2.putText(image, "Hexagon", coords, font, 1, colour, 1)
    else:
        # If the length is not any of the above, we will guess the shape/contour to be a circle.
        cv2.putText(image, "Circle", coords, font, 1, colour, 1)

# Displaying the image with the detected shapes onto the screen
cv2.imshow("shapes_detected", image)
cv2.waitKey(0)



