import cv2
import numpy as np
# from matplotlib import pyplot as plt

# from segments import Segments



# import cv2
# import numpy as np
#
# # load image
# img = cv2.imread("green-font.png");
# # img = cv2.imread("/home/ashish/Downloads/demo-transformed.jpeg");
#
# # crop
# # img = img[300:800, 100:800, :];
#
# # lab
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
# l, a, b = cv2.split(lab);
#
# # show
# cv2.imshow("orig", img);
#
# # closing operation
# kernel = np.ones((2, 2), np.uint8);
#
# # threshold params
# low = 165;
# high = 200;
# iters = 20;
#
# # make copy
# copy = b.copy();
#
# # threshold
# thresh = cv2.inRange(copy, low, high);
# cv2.imwrite("threshold0.jpg", thresh);
#
# # dilate
# for a in range(iters):
#     thresh = cv2.dilate(thresh, kernel);
# cv2.imwrite("threshold3.jpg", thresh);

# # load image
# img = cv2.imread("/home/plutusdev/Downloads/threshold2.png");
#
# # crop
# # img = img[300:800, 100:800, :];
#
# # lab
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
# l, a, b = cv2.split(lab);
#
# # show
# cv2.imshow("orig", img);
#
# # closing operation
# kernel = np.ones((5, 5), np.uint8);
#
# # threshold params
# low = 165;
# high = 200;
# iters = 3;
#
# # make copy
# copy = b.copy();
#
# # threshold
# thresh = cv2.inRange(copy, low, high);
#
# # dilate
# for a in range(iters):
#     thresh = cv2.dilate(thresh, kernel);
#
# # erode
# for a in range(iters):
#     thresh = cv2.erode(thresh, kernel);
#
# # show image
# cv2.imshow("thresh", thresh);
# cv2.imwrite("threshold.jpg", thresh);
#
# # start processing
# _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
#
# # draw
# for contour in contours:
#     cv2.drawContours(img, [contour], 0, (0, 255, 0), 3);
#
# # get res of each number
# bounds = [];
# h, w = img.shape[:2];
# for contour in contours:
#     left = w;
#     right = 0;
#     top = h;
#     bottom = 0;
#     for point in contour:
#         point = point[0];
#         x, y = point;
#         if x < left:
#             left = x;
#         if x > right:
#             right = x;
#         if y < top:
#             top = y;
#         if y > bottom:
#             bottom = y;
#     tl = [left, top];
#     br = [right, bottom];
#     bounds.append([tl, br]);

# crop out each number
# cuts = [];
# number = 0;
# for bound in [[[30,10], [120,217]]]:
#     tl, br = bound;
#     cut_img = thresh[tl[1]:br[1], tl[0]:br[0]];
#     cuts.append(cut_img);
#     number += 1;
#     cv2.imshow(str(number), cut_img);
#
# # font
# font = cv2.FONT_HERSHEY_SIMPLEX;
#
# # create a segment model
# model = Segments();
# index = 0;
# for cut in cuts:
#     # save image
#     cv2.imwrite(str(index) + "_" + str(number) + ".jpg", cut);
#
#     # process
#     model.digest(cut);
#     number = model.getNum();
#     print(number);
#     cv2.imshow(str(index), cut);
#
#     # draw and save again
#     h, w = cut.shape[:2];
#     drawn = np.zeros((h, w, 3), np.uint8);
#     drawn[:, :, 0] = cut;
#     drawn = cv2.putText(drawn, str(number), (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA);
#     cv2.imwrite("drawn" + str(index) + "_" + str(number) + ".jpg", drawn);
#
#     index += 1;
#     # cv2.waitKey(0);
#
# # show
# cv2.imshow("contours", img);
# cv2.imwrite("contours.jpg", img);
# cv2.waitKey(0);

# ======================================================================================================================
# import cv2
# import numpy as np
# from segments import Segments
#
# # # load image
# # img = cv2.imread("green-font.png");
# #
# # # crop
# # # img = img[300:800, 100:800, :];
# #
# # # lab
# # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
# # l, a, b = cv2.split(lab);
# #
# # # show
# # cv2.imshow("orig", img);
# #
# # # closing operation
# # kernel = np.ones((5, 5), np.uint8);
# #
# # # threshold params
# # low = 165;
# # high = 200;
# # iters = 3;
# #
# # # make copy
# # copy = b.copy();
# #
# # # threshold
# # thresh = cv2.inRange(copy, low, high);
# #
# # # dilate
# # for a in range(iters):
# #     thresh = cv2.dilate(thresh, kernel);
# #
# # # erode
# # for a in range(iters):
# #     thresh = cv2.erode(thresh, kernel);
# #
# # # show image
# # cv2.imshow("thresh", thresh);
# # cv2.imwrite("threshold.jpg", thresh);
# #
# # # start processing
# # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
# #
# # # draw
# # for contour in contours:
# #     cv2.drawContours(img, [contour], 0, (0, 255, 0), 3);
# #
# # # get res of each number
# # bounds = [];
# # h, w = img.shape[:2];
# # for contour in contours:
# #     left = w;
# #     right = 0;
# #     top = h;
# #     bottom = 0;
# #     for point in contour:
# #         point = point[0];
# #         x, y = point;
# #         if x < left:
# #             left = x;
# #         if x > right:
# #             right = x;
# #         if y < top:
# #             top = y;
# #         if y > bottom:
# #             bottom = y;
# #     tl = [left, top];
# #     br = [right, bottom];
# #     bounds.append([tl, br]);
#
# # # crop out each number
# # cuts = [];
# # number = 0;
# # for bound in [[0,0],[136,217]]:
# #     tl, br = bound;
# #     cut_img = thresh[tl[1]:br[1], tl[0]:br[0]];
# #     cuts.append(cut_img);
# #     number += 1;
# #     cv2.imshow(str(number), cut_img);
#
# # font
# font = cv2.FONT_HERSHEY_SIMPLEX;
#
# # create a segment model
# model = Segments();
# index = 0;
#
# imgs = cv2.imread("/home/plutusdev/Downloads/threshold2.png")
#
# for cut in [imgs]:
#     # process
#     model.digest(cut);
#     number = model.getNum();
#     print(number);
#     cv2.imshow(str(index), cut);
#
#     # draw and save again
#     h, w = cut.shape[:2];
#     drawn = np.zeros((h, w, 3), np.uint8);
#     drawn[:, :, 0] = cut;
#     drawn = cv2.putText(drawn, str(number), (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA);
#     cv2.imwrite("drawn_threshold2.jpg", drawn);
#
#     index += 1;
#     # cv2.waitKey(0);
#
# # show
# # cv2.imshow("contours", imgs);
# # cv2.imwrite("contours.jpg", img);
# # cv2.waitKey(0);


# ==================================================================================================================


# import pytesseract
# import cv2
#
# image = cv2.imread("threshold1.jpg")
# cv2.imshow("image", image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("green-font-gray.png", gray)
#
# blur =  cv2.GaussianBlur(gray, (7,7), 0)
# cv2.imwrite("green-font-blur.png", blur)
#
# thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imwrite("green-font-thresh.png", thresh)
#
# kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
# cv2.imwrite("green-font-kernal.png", kernal)
#
# dilate = cv2.dilate(thresh, kernal, iterations=1)
# cv2.imwrite("green-font-dilate.png", dilate)
#
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # import pdb; pdb.set_trace()
# cnts = sorted(cnts, key = lambda x: cv2.boundingRect(x)[0])
# for i in cnts:
#     x, y, w, h =cv2.boundingRect(i)
#     print(x,"1")
#     print(y,"2")
#     print(w,"3")
#     print(h ,"4")
#     if h > 150 and w > 150:
#     #
#
#     # import pdb; pdb.set_trace()
#     # cv2.rectangle(image, ([x[1]], [x[0]]), (127, 221), (30, 250, 12), 2)
#
#         # cv2.rectangle(image, (x,y), (x+w, y+h), (30, 250, 12), 2)
#         cv2.rectangle(image, (x,y), (127, 221), (30, 250, 12), 2)
#         cv2.rectangle(image, (137,4), (261, 220), (30, 250, 12), 2)
#         cv2.rectangle(image, (346,12), (391, 212), (30, 250, 12), 2)
#         cv2.rectangle(image, (405,4), (529, 217), (30, 250, 12), 2)
# cv2.imwrite("green-font-bbox.png", image)
# cv2.imshow("green-font-bbox.png", image)
# cv2.waitKey(0)



# =====================================================================================================

#
# #
# import cv2
# import numpy as np
#
# # load image
# img = cv2.imread("green-font.png");
# # img = cv2.imread("/home/ashish/Downloads/demo-transformed.jpeg");
#
# # lab
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
# l, a, b = cv2.split(lab);
#
# # show
# cv2.imshow("orig", img);
#
# # closing operation
# kernel = np.ones((2, 2), np.uint8);
#
# # threshold params
# low = 165;
# high = 200;
# iters = 20;
#
# # make copy
# copy = b.copy();
#
# # threshold
# thresh = cv2.inRange(copy, low, high);
# cv2.imwrite("threshold0.jpg", thresh);
#
# # dilate
# for a in range(iters):
#     thresh = cv2.dilate(thresh, kernel);
# cv2.imwrite("threshold3.jpg", thresh);
#
#
#
img_with_four_numbers = cv2.imread("threshold3.jpg");
# import pdb; pdb.set_trace()
row = len(img_with_four_numbers)
col = len(img_with_four_numbers[0])
from PIL import Image
# cv2.imshow("img",img_with_four_numbers)


def end_point(rows, col):
    fix_row = row - 1
    for k in range(col, 0, -1):
        # import pdb; pdb.set_trace()
        if (str(img_with_four_numbers[fix_row][k]) == '[0 0 0]') and (str(img_with_four_numbers[fix_row][k-1]) == '[255 255 255]'):
            # import pdb; pdb.set_trace()

            return fix_row, k
        if (str(img_with_four_numbers[fix_row][k]) == '[255 255 255]') and (
                str(img_with_four_numbers[fix_row][k - 1]) == '[0 0 0]') and (str(img_with_four_numbers[fix_row][k-2]) == '[0 0 0]'):
            return fix_row, k
def start_point(rows, col):
    fix_row = row - 1
    for k in range(col, 0, -1):
        import pdb; pdb.set_trace()
        if (str(img_with_four_numbers[fix_row][k]) == '[255 255 255]') and (str(img_with_four_numbers[fix_row][k-1]) == '[0 0 0]'):
            import pdb; pdb.set_trace()
            print("this is 2nd check point...............!")
            return fix_row, k


imgs = cv2.imread("threshold3.jpg")

gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
cv2.imwrite("green-font-gray.png", gray)

blur =  cv2.GaussianBlur(gray, (7,7), 0)
cv2.imwrite("green-font-blur.png", blur)

thre = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("green-font-thresh.png", thre)

kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
cv2.imwrite("green-font-kernal.png", kernal)

dilate = cv2.dilate(thre, kernal, iterations=1)
cv2.imwrite("green-font-dilate.png", dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# import pdb; pdb.set_trace()
cnts = sorted(cnts, key = lambda x: cv2.boundingRect(x)[0])

after_first_box_ending_point =[]
flage = True
crop_name = 0
for i in cnts:
    x, y, w, h =cv2.boundingRect(i)
    for i in range(row, 0, -1):
        for j in range(col, 0, -1):
            # import pdb;pdb.set_trace()
            if str(img_with_four_numbers[i-1][j-1]) == '[255 255 255]':
                # row-=1
                # col-=1
                # import pdb;pdb.set_trace()
                aa = list(end_point(row-1, j-1))

                if flage:
                    cv2.rectangle(img_with_four_numbers, (aa[1], 0), (j,row), (30, 250, 12), 1)
                    cv2.imwrite("green-font-bbox.png", img_with_four_numbers)

                    img_with_four_numbers = Image.open('green-font-bbox.png')

                    crop_img = img_with_four_numbers.crop((aa[1]+1, 1, j+1, row+1))
                    crop_img.save("last_image.jpg")
                    crop_img.show()
                    img_with_four_numbers = np.array(img_with_four_numbers)

                    after_first_box_ending_point.append(aa[1]-1)
                    aa.clear()
                    flage = False
                else:
                     crop_name += 1
                     cv2.rectangle(img_with_four_numbers, (aa[1], 0), (after_first_box_ending_point[0],row), (30, 250, 12), 1)
                     cv2.imwrite("green-font-bbox.png", img_with_four_numbers)
                     img_with_four_numbers = im = Image.open('green-font-bbox.png')
                     # import pdb; pdb.set_trace()
                     crop_img = img_with_four_numbers.crop((aa[1]+1, 1, after_first_box_ending_point[0]+1, row+1))
                     crop_img.save(f"last_image{str(crop_name)}.jpg")

                     crop_img.show()
                     img_with_four_numbers = np.array(img_with_four_numbers)
                     # img_with_four_numbers = img_with_four_numbers.crop((after_first_box_ending_point[0], 0, j, row))

                     after_first_box_ending_point.clear()
                     print(after_first_box_ending_point, "after_first_box_ending_point")
                     after_first_box_ending_point.append(aa[1] - 1)
                     aa.clear()
                    # else:
                    #     import pdb; pdb.set_trace()
                    #     bb =list(start_point(row-1, j-1))
                    #     import pdb; pdb.set_trace()

                    # cv2.rectangle(img_with_four_numbers, (aa[1], 0), (after_first_box_ending_point[0], row),
                    #              (30, 250, 12), 1)
                    # cv2.imwrite("green-font-bbox.png", img_with_four_numbers)
                    # img_with_four_numbers = im = Image.open('green-font-bbox.png')
                    # # import pdb; pdb.set_trace()
                    # crop_img = img_with_four_numbers.crop(
                    #     (aa[1] + 1, 1, after_first_box_ending_point[0] + 1, row + 1))
                    # crop_img.save(f"last_image{str(crop_name)}.jpg")
                    #
                    # crop_img.show()
                    # img_with_four_numbers = np.array(img_with_four_numbers)
                    # ...


                cv2.imwrite("green-font-bbox.png", img_with_four_numbers)
                cv2.imshow("green-font-bbox.png", img_with_four_numbers)
                cv2.waitKey(0)
                cv2.destroyAllWindows()




# import cv2
#
# # read the input image
# # img = img1 = cv2.imread('green-font-thresh.jpg')
# img = img1 = cv2.imread('threshold3.jpg')
#
# for i in range(len(img)):
#     for j in range(len(img[0])):
#         if str(img[i][j]) == '[255 255 255]':
#             img[i][j] = [0,0,0]
#         elif str(img[i][j]) == '[0 0 0]':
#             img[i][j] = [255, 255, 255]
#         else:
#             img[i][j] = [255, 255, 255]
#
# cv2.imwrite('my_img2.jpeg', img)
#
#
# # convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # apply thresholding on the gray image to create a binary image
# ret,thresh = cv2.threshold(gray,127,255,0)
#
# # find the contours
# contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# # take the first contour
# cnt = contours[0]
#
# # compute the bounding rectangle of the contour
# x,y,w,h = cv2.boundingRect(cnt)
#
# # draw contour
# img = cv2.drawContours(img,[cnt],0,(0,255,255),2)
#
# # draw the bounding rectangle
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#
# # display the image with bounding rectangle drawn on it
# cv2.imshow("Bounding Rectangle", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2  # Computer vision library

# # Read the color image
# image = cv2.imread("my_img2.jpeg")
# # image = cv2.imread("threshold3.jpg")
# # image = cv2.imread("red-font.png")
#
# # Make a copy
# new_image = image.copy()
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Display the grayscale image
# cv2.imshow('Gray image', gray)
# cv2.waitKey(0)  # Wait for keypress to continue
# cv2.destroyAllWindows()  # Close windows
#
# # Convert the grayscale image to binary
# ret, binary = cv2.threshold(gray, 100, 255,
#                             cv2.THRESH_OTSU)
#
# # Display the binary image
# cv2.imshow('Binary image', binary)
# cv2.waitKey(0)  # Wait for keypress to continue
# cv2.destroyAllWindows()  # Close windows
#
# # To detect object contours, we want a black background and a white
# # foreground, so we invert the image (i.e. 255 - pixel value)
# inverted_binary = ~binary
# cv2.imshow('Inverted binary image', inverted_binary)
# cv2.waitKey(0)  # Wait for keypress to continue
# cv2.destroyAllWindows()  # Close windows
#
# # Find the contours on the inverted binary image, and store them in a list
# # Contours are drawn around white blobs.
# # hierarchy variable contains info on the relationship between the contours
# contours, hierarchy = cv2.findContours(inverted_binary,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# # Draw the contours (in red) on the original image and display the result
# # Input color code is in BGR (blue, green, red) format
# # -1 means to draw all contours
# print(contours, "this is contours ................!")
# with_contours = cv2.drawContours(image, contours, -1, (255, 0, 255), 3)
#
# cv2.imshow('Detected contours', with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Show the total number of contours that were detected
# print('Total number of contours detected: ' + str(len(contours)))
#
# # Draw just the first contour
# # The 0 means to draw the first contour
# first_contour = cv2.drawContours(new_image, contours, 0, (255, 0, 255), 3)
# cv2.imshow('First detected contour', first_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Draw a bounding box around the first contour
# # x is the starting x coordinate of the bounding box
# # y is the starting y coordinate of the bounding box
# # w is the width of the bounding box
# # h is the height of the bounding box
# x, y, w, h = cv2.boundingRect(contours[0])
# cv2.rectangle(first_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)
# cv2.imshow('First contour with bounding box', first_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Draw a bounding box around all contours
# for c in contours:
#     # import pdb;pdb.set_trace()
#     x, y, w, h = cv2.boundingRect(c)
#
#     # Make sure contour area is large enough
#     if (cv2.contourArea(c)) > 10:
#         cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         # cropped_image = with_contours[y:y + h, x:x + w]
#         # plt.imshow(cropped_image)
#         # cv2.imshow(cropped_image,'contour1' )
#         # cv2.imwrite('contour1.png', cropped_image)
#
# cv2.imshow('All contours with bounding box', with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()