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

#
# import cv2  # OpenCV Library
#
# # Image to detect shapes on below
# image = cv2.imread("SaleDigit.jpg")
#
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to gray image
#
# # Setting threshold value to get new image (In simpler terms: this function checks every pixel, and depending on how
# # dark the pixel is, the threshold value will convert the pixel to either black or white (0 or 1)).
# _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
#
# # Retrieving outer-edge coordinates in the new threshold image
# contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Iterating through each contour to retrieve coordinates of each shape
# for i, contour in enumerate(contours):
#     if i == 0:
#         continue
#
#     # The 2 lines below this comment will approximate the shape we want. The reason being that in certain cases the
#     # shape we want might have flaws or might be imperfect, and so, for example, if we have a rectangle with a
#     # small piece missing, the program will still count it as a rectangle. The epsilon value will specify the
#     # precision in which we approximate our shape.
#     epsilon = 0.01 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#
#     # Drawing the outer-edges onto the image
#     cv2.drawContours(image, contour, 0, (0, 0, 0), 4)
#
#     # Retrieving coordinates of the contour so that we can put text over the shape.
#     x, y, w, h = cv2.boundingRect(approx)
#     x_mid = int(x + (w / 3))  # This is an estimation of where the middle of the shape is in terms of the x-axis.
#     y_mid = int(y + (h / 1.5))  # This is an estimation of where the middle of the shape is in terms of the y-axis.
#
#     # Setting some variables which will be used to display text on the final image
#     coords = (x_mid, y_mid)
#     colour = (0, 0, 0)
#     font = cv2.FONT_HERSHEY_DUPLEX
#
#     # This is the part where we actually guess which shape we have detected. The program will look at the amount of edges
#     # the contour/shape has, and then based on that result the program will guess the shape (for example, if it has 3 edges
#     # then the chances that the shape is a triangle are very good.)
#     #
#     # You can add more shapes if you want by checking more lenghts, but for the simplicity of this tutorial program I
#     # have decided to only detect 5 shapes.
#     if len(approx) == 3:
#         cv2.putText(image, "Triangle", coords, font, 1, colour, 1)  # Text on the image
#     elif len(approx) == 4:
#         cv2.putText(image, "Quadrilateral", coords, font, 1, colour, 1)
#     elif len(approx) == 5:
#         cv2.putText(image, "Pentagon", coords, font, 1, colour, 1)
#     elif len(approx) == 6:
#         cv2.putText(image, "Hexagon", coords, font, 1, colour, 1)
#     else:
#         # If the length is not any of the above, we will guess the shape/contour to be a circle.
#         cv2.putText(image, "Circle", coords, font, 1, colour, 1)
#
# # Displaying the image with the detected shapes onto the screen
# cv2.imshow("shapes_detected", image)
# cv2.waitKey(0)







#
# import cv2
# import numpy as np
# from PIL import Image
# img = cv2.imread("image3.png");
# # img = cv2.imread("green-font.png");
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("green-font-gray.png", gray)
#
# blur =  cv2.GaussianBlur(gray, (7,7), 0)
# cv2.imwrite("green-font-blur.png", blur)
#
# img = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imwrite("green-font-thresh.png", img)
# row = len(img)
# col = len(img[0])
#
#
#
#
#
# def find_first_black_row(row, col):
#     for first_black in range(0 , row-1):
#         for first_col_black in range(0, col-1):
#             if int(img[first_black][first_col_black]) == 0:
#                 # import pdb; pdb.set_trace()
#                 print("this is black point", first_black, first_col_black)
#                 return first_black
#     return find_first_black_row()
#
# def find_last_black_row(row, col):
#     for last_row_black in range(row-1, 0, -1):
#         for last_col_black in range(0, col-1):
#             if int(img[last_row_black][last_col_black]) == 0:
#                 # import pdb; pdb.set_trace()
#                 print("this is black point", last_row_black, last_col_black)
#                 return last_row_black
#     return find_last_black_row()
#
#
#
# def afterfirst_ending_point(i, j, aa):
#     fix_row = row - 1
#     # import pdb; pdb.set_trace()
#     for j in range(j - 1, 0, -1):
#         for i in range(fix_row, 0, -1):
#             if int(img[i][j]) == 0:
#                 if (j + 1) in aa:
#                     return i, j
#                 print("this is start point")
#     return afterfirst_ending_point()
#
# def start_point(i, j, point):
#     fix_row = row - 1
#     for j in range(j-1, 0, -1):
#         for i in range(fix_row, 0, -1):
#             if int(img[i][j]) == 0:
#                 if (j-1) in point:
#                     return i, j
#                 print("this is start point")
#     return start_point()
# def end_point(rows, col):
#     fix_row = row - 1
#     check_point = False
#     point = []
#     white_pixel_check =[]
#     for j in range(col, 0, -1):
#         for i in range(fix_row, 0, -1):
#
#             white_pixel_check.append((int(img[i][j])))
#             if int(img[i][j]) == 0:
#                 break
#         white_pixel_check.sort()
#         if white_pixel_check[0] == 255:
#             point.append(j)
#             white_pixel_check.clear()
#         else:
#             white_pixel_check.clear()
#
#     return point
#
# row_cut_start_point = find_first_black_row(row, col)
# row_cut_end_point = find_last_black_row(row, col)
# after_first_box_ending_point =[]
# check_start_point =[]
# flage = True
# start_point1 =[]
# crop_name = 0
# for j in range(col, 0, -1):
#     for i in range(row, 0, -1):
#         if int(img[i - 1][j - 1]) == 0:
#             aa = list(end_point(i-1, j - 1))
#             start_point1 = list(start_point(i-1, j-1,aa))
#             print("this is check point...........!")
#             if flage:
#                 # import pdb; pdb.set_trace() n]

#                 cv2.rectangle(img, (start_point1[1], row_cut_start_point), (j, row_cut_end_point), (30, 250, 12), 1)
#                 cv2.imwrite("green-font-bbox.png", img)
#                 img = Image.open('green-font-bbox.png')
#                 crop_img = img.crop((start_point1[1] + 1, row_cut_start_point, j, row_cut_end_point))
#                 crop_img.save("last_image.jpg")
#                 crop_img.show()
#                 img = np.array(img)
#                 after_first_box_ending_point.append(start_point1[1] - 1)
#                 start_point1.clear()
#                 flage = False
#             else:
#                 crop_name += 1
#                 ending_point = list(afterfirst_ending_point(i, j, aa))
#                 # import pdb;pdb.set_trace()
#                 cv2.rectangle(img, (start_point1[1], row_cut_start_point), (ending_point[1], row_cut_end_point), (30, 250, 12),1)
#                 cv2.imwrite("green-font-bbox.png", img)
#                 img = Image.open('green-font-bbox.png')
#                 crop_img = img.crop((start_point1[1] + 1, row_cut_start_point, ending_point[1] + 1,row_cut_end_point))
#                 crop_img.save(f"last_image{str(crop_name)}.jpg")
#                 crop_img.show()
#                 img = np.array(img)
#                 after_first_box_ending_point.clear()
#                 print(after_first_box_ending_point, "after_first_box_ending_point")
#                 after_first_box_ending_point.append(start_point1[1] - 1)
#                 start_point1.clear()
#                 ending_point.clear()



# img = cv2.imread("download (2).png");
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("font/image17.jpg", gray)


# =============================================================================
"""
                  CONVERT IMAGES INTO THRESHOLD
                  """
# import cv2;
# import numpy as np;
#
#
# lst = ['/home/plutusdev/Projects/font-detector-main/1677488820914.JPEG','/home/plutusdev/Projects/font-detector-main/1677488850080.JPEG','/home/plutusdev/Projects/font-detector-main/1677488861789.JPEG','/home/plutusdev/Projects/font-detector-main/1677488864549.bmp','/home/plutusdev/Projects/font-detector-main/1677488864549.JPEG']
# for img in lst:
#       name = img.split('/')[-1].split('.')[0]
#       # Read image
#       im_in = cv2.imread(img);
#       im_in = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY);
#
#
#       # Threshold.
#       # Set values efont-detection (1)qual to or above 50 to 0.
#       # Set values below 50 to 255.
#       th, im_th = cv2.threshold(im_in, 35, 255, cv2.THRESH_BINARY_INV);
#
#
#       # Copy the thresholded image.
#       im_floodfill = im_th.copy()
#
#
#       # Mask used to flood filling.
#       # Notice the size needs to be 2 pixels than the image.
#       h, w = im_th.shape[:2]
#       mask = np.zeros((h + 2, w + 2), np.uint8)
#
#
#       # Floodfill from point (0, 0)
#       cv2.floodFill(im_floodfill, mask, (0, 0), 255);
#
#
#       # Invert floodfilled image
#       im_floodfill_inv = thresh = cv2.bitwise_not(im_floodfill)
#       cv2.imwrite('white_'+name+'.jpg', im_floodfill_inv)

# =============================================================================================================


# from PIL import Image
#
# # INPUT_IMAGE_URL = "https://www.shutterstock.com/image-vector/machinery-downtime-maintenance-led-controller-600w-1026664582.jpg" #@param {type:"string"}
# DETECTION_THRESHOLD = 0.3 #@param {type:"number"}
# # TFLITE_MODEL_PATH = "font-detection.tflite"
#
# # TEMP_FILE = '/tmp/image.png'
#
# # TEMP_FILE = cv2.imread("image4.png")
# image = Image.open("/home/plutusdev/Downloads/1677222617120.jpeg").convert('RGB')
# image.thumbnail((512, 512), Image.ANTIALIAS)
# image_np = np.asarray(image)
#
#
# im_in = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY);
# cv2.imshow("img",im_in)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Threshold.
# # Set values equal to or above 50 to 0.
# # Set values below 50 to 255.
# th, im_th = cv2.threshold(im_in, 50, 255, cv2.THRESH_BINARY_INV);
#
#
# # Copy the thresholded image.
# im_floodfill = im_th.copy()
#
#
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = im_th.shape[:2]
# mask = np.zeros((h + 2, w + 2), np.uint8)
#
#
# # Floodfill from point (0, 0)
# # cv2.floodFill(im_floodfill, mask, (0, 0), 255);
#
#
# # Invert floodfilled image
# image_np = thresh = cv2.bitwise_not(im_floodfill)
# image_np = cv2.threshold(im_floodfill, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# for i in range(len(image_np)):
#   for j in range(len(image_np[0])):
#     if int(image_np[i][j]) == 0:
#       image_np[i][j] = np.uint8(255)
#     else:
#       image_np[i][j] = np.uint8(0)

# black_pix = np.where((image_np == [0, 0, 0]).all(axis=2))
# white_pix = np.where((image_np == [255, 255, 255]).all(axis=2))

# image_np[black_pix] = [255, 255, 255]
# image_np[white_pix] = [0, 0, 0]


# cv2.imshow('my_img.jpg', image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('my_img.jpg', image_np)
# image = Image.open('my_img.jpg').convert('RGB')
# image.show()
# image_np = np.asarray(image)
#
# # Load the TFLite model
# options = ObjectDetectorOptions(
#       num_threads=4,
#       score_threshold=DETECTION_THRESHOLD,
# )
# detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)
#
# # Run object detection estimation using the model.
# detections = detector.detect(image_np)
# print(detections[0][1][0].label)
# print(detections)
# print(type(detections[0][1][0]))
#
#
# # Draw keypoints and edges on input image
# image_np = visualize(image_np, detections)
#
# # Show the detection result
# Image.fromarray(image_np)
















# ====================================================================================


import numpy as np
import glob
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
# %matplotlib widget

# load the image
image = iio.imread("1677488850080.JPEG")

fig, ax = plt.subplots()
plt.imshow(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_images", gray_image)
cv2.waitKey(0)
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap="gray")
cv2.imshow("blurred_image", blurred_image)
cv2.waitKey(0)


t = 0.4
binary_mask = blurred_image < t
iio.imwrite('astronaut-gray.jpg', binary_mask)
cv2.imshow("binary_mask", binary_mask)
cv2.waitKey(0)

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap="gray")
