import cv2
import pytesseract
# #
pytesseract.pytesseract.tesseract_cmd = "/bin/tesseract"
# # #
# image = cv2.imread('0-9-on-seven-segment.png', 0)
# # import pdb; pdb.set_trace()
# edges = cv2.Canny(image,200,300,True)
#
# thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# #
# data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
# # data1 = pytesseract.image_to_string(edges, lang='eng', config='--psm 6')
# print(data)
# # import pdb; pdb.set_trace()
# # print('------',data1, "-----")
# cv2.imshow("Edge Detected Image", edges)
#
# cv2.imshow('thresh', thresh)
# #
# cv2.waitKey()
#



# # # import cv2
import numpy as np

# load image
# img = cv2.imread("/home/ashish/Downloads/green-font.png");
img = cv2.imread("green-font.png");
# img = cv2.imread("0-9-on-seven-segment.png");

# crop
# img = img[300:800, 100:800, :];

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
iters = 17;

# make copy
copy = b.copy();

# threshold
thresh = cv2.inRange(copy, low, high);
cv2.imwrite("threshold0.jpg", thresh);


for a in range(iters):
    thresh = cv2.dilate(thresh, kernel);
cv2.imwrite("threshold1.jpg", thresh);

# data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 10')
# # data1 = pytesseract.image_to_string(edges, lang='eng', config='--psm 6')
# print(data)
# # import pdb; pdb.set_trace()
# # print('------',data1, "-----")
# # cv2.imshow("Edge Detected Image", edges)
#
# cv2.imshow('thresh', thresh)
#
# cv2.waitKey()
#




# import sys
#
# import numpy as np
# import cv2
#
# im = cv2.imread('threshold1.jpg')
# # im3 = im.copy()
#
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# samples = np.empty((0, 100))
# responses = []
# keys = [i for i in range(48, 58)]
#
# for cnt in contours:
#     if cv2.contourArea(cnt) > 50:
#         [x, y, w, h] = cv2.boundingRect(cnt)
#
#         if h > 28:
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             roi = thresh[y:y + h, x:x + w]
#             roismall = cv2.resize(roi, (10, 10))
#             cv2.imshow('norm', im)
#             key = cv2.waitKey(0)
#
#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1, 100))
#                 samples = np.append(samples, sample, 0)
#
# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size, 1))
# print("training complete")
#
# np.savetxt('generalsamples.data', samples)
# np.savetxt('generalresponses.data', responses)



