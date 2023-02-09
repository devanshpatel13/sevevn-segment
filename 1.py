import cv2
import pytesseract
# import pdb

# pdb.set_trace()
pytesseract.pytesseract.tesseract_cmd = '/bin/tesseract'
img = cv2.imread("/home/rajkplutus/PycharmProjects/text-detector/red-font.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

abc = pytesseract.image_to_string(img)
# print(int(abc))
# print(len(abc))
file = open("green.txt", "w+")
file.write(abc)
file.close()
cv2.imshow('result', img)
cv2.waitKey(0)

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