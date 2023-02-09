import cv2
from PIL import Image
from io import StringIO
from pytesseract import pytesseract

# import pdb

# pdb.set_trace()
camera = cv2.VideoCapture(0)
while True:
    _, image = camera.read()
    cv2.imshow('text_detect', image)
    # 0xFF is just used to mask off the last 8bits of the sequence
    # the ord() of any english keyboard character will not be greater than 255
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('text.jpg', image)
        break
camera.release()
cv2.destroyAllWindows()


def tesseract_demo():
    # import pdb
    # pdb.set_trace()
    path = r'/bin/tesseract'
    img_path = "text.jpg"
    pytesseract.tesseract_cmd = path
    # for string and digit font detector.
    # text = pytesseract.image_to_string(Image.open(img_path))
    # only digit detector.
    text = pytesseract.image_to_string(img_path, config='digits')
    print(text[:-1])


tesseract_demo()
