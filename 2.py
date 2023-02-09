# import cv2
# from PIL import Image
# from pytesseract import pytesseract
#
# camera = cv2.VideoCapture(0)
# while True:
#     _, image = camera.read()
#     cv2.imshow('text_detect', image)
#     # 0xFF is just used to mask off the last 8bits of the sequence
#     # the ord() of any english keyboard character will not be greater than 255.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.imwrite('a.jpg', image)
#         break
# camera.release()
# cv2.destroyAllWindows()
#
#
# def demo():
#     path = r'/bin/tesseract'
#     img = 'a.jpg'
#     pytesseract.tesseract_cmd = path
#     text = pytesseract.image_to_string(Image.open(img))
#     print(text)
# demo()
#
#
#
import cv2
from PIL import Image
from pytesseract import pytesseract

camera = cv2.VideoCapture(0)
cv2.namedWindow("text-detect")
img_counter = 0


# import cv2
# import pytesseract
# import pdb
# # pdb.set_trace()
# # if windows
# pytesseract.pytesseract.tesseract_cmd = '/bin/tesseract'
#
# img = cv2.imread('green-font.png', 0)
# img = cv2.resize(img, (0, 0), fx=2, fy=2)
#
# config = ("digits")
#
# data = pytesseract.image_to_string(img, lang='eng', config=config)
#
# print(data)

########################################
# from imutils.perspective import four_point_transform
# from imutils import contours
# import imutils
# import cv2
#
# # creating a dictionary for 7-segment detection
# DIGITS_LOOKUP = {
#     (1, 1, 1, 0, 1, 1, 1): 0,
#     (0, 0, 1, 0, 0, 1, 0): 1,
#     (1, 0, 1, 1, 1, 1, 0): 2,
#     (1, 0, 1, 1, 0, 1, 1): 3,
#     (0, 1, 1, 1, 0, 1, 0): 4,
#     (1, 1, 0, 1, 0, 1, 1): 5,
#     (1, 1, 0, 1, 1, 1, 1): 6,
#     (1, 0, 1, 0, 0, 1, 0): 7,
#     (1, 1, 1, 1, 1, 1, 1): 8,
#     (1, 1, 1, 1, 0, 1, 1): 9
# }
#
# # capturing from webcam
# cap = cv2.VideoCapture(0)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
#
# # define codec and create VideoWriter object
# out = cv2.VideoWriter('out_videos/cam_blur.avi',
#                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
#                       30,
#                       (frame_width, frame_height))
#
# # continuous capture from webcam
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#         edged = cv2.Canny(blurred, 50, 200, 200)
#         cv2.imshow('Video', edged)
#
#         # find contours in the edge map, then sort them by their size in descending order
#         cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#         displayCnt = None
#
#         # loop over the contours
#         for c in cnts:
#             # approximate the contour
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#             # if the contour has four vertices, then we have found
#             # the thermostat display
#             if len(approx) == 4:
#                 displayCnt = None
#                 break
#
#         # extract the thermostat display, apply a perspective transform
#         # to it
#         warped = four_point_transform(gray, displayCnt.reshape(4, 2))
#         output = four_point_transform(frame, displayCnt.reshape(4, 2))
#         cv2.imshow('Birdeye', output)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

##########################################
# import cv2
# import pytesseract
#
#
# def detect_7_segments(image):
#     config = '--psm 6 -c tessedit_char_whitelist="0123456789"'
#     return pytesseract.image_to_string(image, config=config)
#
#
# for img_file in ['green-font.png', 'red-font.png']:
#     img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
#     img = cv2.blur(img, (5, 5))
#     img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)[1]
#     print(img_file, detect_7_segments(img).replace('\f', '').replace('\n', ''))
#
# import os
#
# import matplotlib.image as mpimg
# from matplotlib import pyplot as plt

# for img in os.listdir("/home/rajkplutus/PycharmProjects/text-detector"):
#     test_NP = mpimg.imread("green-font.png")

# plt.imshow(test_NP)
# plt.axis('off')
# plt.title('font-detection')
# plt.show()
#
