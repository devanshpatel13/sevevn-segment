from segments import Segments
import cv2
import numpy as np

# load image
img = cv2.imread("green-font.png");
# img = cv2.imread("/home/ashish/Downloads/demo-transformed.jpeg");

# lab
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
l, a, b = cv2.split(lab);

# show
cv2.imshow("orig", img);

# closing operation
kernel = np.ones((2, 2), np.uint8);

# threshold params
low = 165;
high = 200;
iters = 18;

# import pdb; pdb.set_trace()
# make copy
copy = b.copy();

# threshold
thresh = cv2.inRange(copy, low, high);
cv2.imwrite("threshold0.jpg", thresh);

# dilate
for a in range(iters):
    thresh = cv2.dilate(thresh, kernel);
cv2.imwrite("threshold3.jpg", thresh);
cv2.imshow("threshold3", thresh)

img_with_four_numbers = cv2.imread("threshold1.jpg");
row = len(img_with_four_numbers)
col = len(img_with_four_numbers[0])


# for i in range(row-1, 0, -1):
#     for j in range(col-1, 0, -1):
#         import pdb;pdb.set_trace()
#         if str(img_with_four_numbers[i][j]) == '[255 255 255]':
#             cv2.rectangle(img_with_four_numbers, (i, j), (127, 221), (30, 250, 12), 2)
#
#             if str(img_with_four_numbers[i][j]) == '[0 0 0]' and \
#                     str(img_with_four_numbers[i+1][j+1]) == '[255 255 255]':
#
#                 cv2.rectangle(img_with_four_numbers, (i, j), (127, 221), (30, 250, 12), 2)




# # ==================================
# # crop out each number
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
