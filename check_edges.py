import cv2
import pytesseract


img = cv2.imread("last_image.jpg")
# cv2.imshow("hhh", img)
import pdb; pdb.set_trace()
# cv2.waitKey()
segments =[]

row = len(img)
col = len(img[0])

if str(img[0][col//2]) == '[255 255 255]':
    segments.append(1)

if str(img[row//4][col-1]) == '[255 255 255]':
    segments.append(2)

if str(img[row//4][0]) == '[255 255 255]':
    segments.append(3)

if str(img[row//2][col//2]) == '[255 255 255]':
    segments.append(4)

if str(img[row*3//4][0]) == '[255 255 255]':
    segments.append(5)

if str(img[row-1][col//2]) == '[255 255 255]':
    segments.append(6)

if str(img[row*3//4][col-1]) == '[255 255 255]':
    segments.append(7)

if [2,7] == segments:
    print('The number is', 1)

if [1, 2, 4, 5, 6] == segments:
    print('The number is', 2)

if [1,2,4,6,7] == segments:
    print('The number is', 3)

if [2,3,4,7] == segments:
    print('The number is', 4)

if [1,3,4,6,7] == segments:
    print('The number is', 5)

if [1,3,4,5,6,7] == segments:
    print('The number is', 6)

if [1,2,7] == segments:
    print('The number is', 7)

if [1,2,3,4,5,6,7] == segments:
    print('The number is', 8)

if [1,2,3,4,6,7] == segments:
    print('The number is', 9)

if [1,2,3,5,6,7] == segments:
    print('The number is', 0)





