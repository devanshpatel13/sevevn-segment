import cv2
# import pytesseract


img = cv2.imread("last_image1.jpg")
# cv2.imshow("hhh", img)
# import pdb; pdb.set_trace()
# cv2.waitKey()
segments =[]

row = len(img)-1
col = len(img[0])-1
flag_data = False
for i in range(0, row):
    if flag_data:
        break
    for j in range(0, col):
        # import pdb; pdb.set_trace()
        if str(img[i][j]) == '[255 255 255]':
            # import pdb; pdb.set_trace()
            # if str(img[i+10][col//2]) == '[255 255 255]':
            if str(img[i+10][col//2]) == '[0 0 0]':
                segments.append(1)
            # import pdb;pdb.set_trace()
            if str(img[row//4][col-5]) !='[255 255 255]':
                segments.append(2)

            # if str(img[row//4][0]) == '[255 255 255]':
            if str(img[row//4][0]) == '[0 0 0]':
                segments.append(3)

            if str(img[row//2][col//2]) == '[0 0 0]':
            # if str(img[row//2][col//2]) == '[255 255 255]':
                segments.append(4)

            # if str(img[row*3//4][0]) == '[255 255 255]':
            if str(img[row*3//4][0]) == '[0 0 0]':
                segments.append(5)

            # if str(img[row-1][col//2]) == '[255 255 255]':
            # if str(img[row-10][col//2]) == '[0 0 0]':
            if str(img[row-10][col//2]) != '[255 255 255]':
                segments.append(6)
            # import pdb; pdb.set_trace()
            # if str(img[row*3//4][col-5]) == '[255 255 255]':
            if str(img[row*3//4][col-5]) == '[0 0 0]':
                segments.append(7)
            print(segments,"this is segment value.............!")
            import pdb;pdb.set_trace()
            if [2,3,5,7] == segments:
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
            flag_data = True
            break


#==========================================================
# segments =[]
#
# row = len(img)
# col = len(img[0])
# import pdb; pdb.set_trace()
# if str(img[0][col//2]) == '[255 255 255]':
#     segments.append(1)
#
# if str(img[row//4][col-1]) == '[255 255 255]':
#     segments.append(2)
#
# if str(img[row//4][0]) == '[255 255 255]':
#     segments.append(3)
#
# if str(img[row//2][col//2]) == '[255 255 255]':
#     segments.append(4)
#
# if str(img[row*3//4][0]) == '[255 255 255]':
#     segments.append(5)
#
# if str(img[row-1][col//2]) == '[255 255 255]':
#     segments.append(6)
#
# if str(img[row*3//4][col-1]) == '[255 255 255]':
#     segments.append(7)
#
# if [2,7] == segments:
#     print('The number is', 1)
#
# if [1, 2, 4, 5, 6] == segments:
#     print('The number is', 2)
#
# if [1,2,4,6,7] == segments:
#     print('The number is', 3)
#
# if [2,3,4,7] == segments:
#     print('The number is', 4)
#
# if [1,3,4,6,7] == segments:
#     print('The number is', 5)
#
# if [1,3,4,5,6,7] == segments:
#     print('The number is', 6)
#
# if [1,2,7] == segments:
#     print('The number is', 7)
#
# if [1,2,3,4,5,6,7] == segments:
#     print('The number is', 8)
#
# if [1,2,3,4,6,7] == segments:
#     print('The number is', 9)
#
# if [1,2,3,5,6,7] == segments:
#     print('The number is', 0)