import cv2
import numpy as np
from PIL import Image
img = cv2.imread("test1.png");
# img = cv2.imread("green-font.png");

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("green-font-gray.png", gray)

blur =  cv2.GaussianBlur(gray, (7,7), 0)
cv2.imwrite("green-font-blur.png", blur)

img = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("green-font-thresh.png", img)
import pdb; pdb.set_trace()
row = len(img)
col = len(img[0])




def afterfirst_ending_point(i, j, aa):
    fix_row = row - 1
    # import pdb; pdb.set_trace()
    for j in range(j - 1, 0, -1):
        for i in range(fix_row, 0, -1):
            if int(img[i][j]) == 0:
                if (j + 1) in aa:
                    return i, j
                # import pdb; pdb.set_trace()
                # ...
                # import pdb; pdb.set_trace()
                print("this is start point")
    return afterfirst_ending_point()

def start_point(i, j, point):
    fix_row = row - 1
    # import pdb; pdb.set_trace()
    for j in range(j-1, 0, -1):
        for i in range(fix_row, 0, -1):
            if int(img[i][j]) == 0:
                if (j-1) in point:
                    return i, j
                # import pdb; pdb.set_trace()
                # ...
                # import pdb; pdb.set_trace()
                print("this is start point")
    return start_point()
def end_point(rows, col):
    fix_row = row - 1
    check_point = False
    point = []
    white_pixel_check =[]
    # import pdb; pdb.set_trace()
    for j in range(col, 0, -1):
        for i in range(fix_row, 0, -1):

            white_pixel_check.append((int(img[i][j])))
            if int(img[i][j]) == 0:
                break
            # if int(img[i][j]) == 0:
            #     point.append(j)
            #     check_point = True
            #     break
        white_pixel_check.sort()
        if white_pixel_check[0] == 255:
            point.append(j)
            white_pixel_check.clear()
        else:
            white_pixel_check.clear()
        # import pdb; pdb.set_trace()

    return point


after_first_box_ending_point =[]
check_start_point =[]
flage = True
start_point1 =[]
crop_name = 0
for j in range(col, 0, -1):
    for i in range(row, 0, -1):
        import pdb; pdb.set_trace()
        if int(img[i - 1][j - 1]) == 0:
            # if int(img[i-2][j-2]) == 255:

            # row-=1
            # col-=1
            import pdb;pdb.set_trace()
            aa = list(end_point(i-1, j - 1))
            start_point1 = list(start_point(i-1, j-1,aa))

            # import pdb;pdb.set_trace()
            print("this is check point...........!")
            if flage:
                # import pdb; pdb.set_trace()
                cv2.rectangle(img, (start_point1[1], 5), (j, i), (30, 250, 12), 1)
                cv2.imwrite("green-font-bbox.png", img)
                img = Image.open('green-font-bbox.png')
                crop_img = img.crop((start_point1[1] + 1, 1, j, i))
                crop_img.save("last_image.jpg")
                crop_img.show()
                img = np.array(img)
                after_first_box_ending_point.append(start_point1[1] - 1)
                start_point1.clear()
                flage = False
            else:
                crop_name += 1
                ending_point = list(afterfirst_ending_point(i, j, aa))
                import pdb;pdb.set_trace()
                cv2.rectangle(img, (start_point1[1], 0), (ending_point[1], row), (30, 250, 12),1)
                # cv2.rectangle(img, (start_point1[1], 0), (after_first_box_ending_point[0], row), (30, 250, 12),1)
                cv2.imwrite("green-font-bbox.png", img)
                img = Image.open('green-font-bbox.png')
                # import pdb; pdb.set_trace()
                crop_img = img.crop((start_point1[1] + 1, 1, ending_point[1] + 1, row + 1))
                # crop_img = img.crop((start_point1[1] + 1, 1, after_first_box_ending_point[0] + 1, row + 1))
                crop_img.save(f"last_image{str(crop_name)}.jpg")
                crop_img.show()
                img = np.array(img)
                # img = img.crop((after_first_box_ending_point[0], 0, j, row))
                after_first_box_ending_point.clear()
                print(after_first_box_ending_point, "after_first_box_ending_point")
                after_first_box_ending_point.append(start_point1[1] - 1)
                start_point1.clear()
                ending_point.clear()