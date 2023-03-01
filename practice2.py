# import cv2
# import numpy as np
# from PIL import Image
# img = cv2.imread("image2.png");
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
# # import pdb; pdb.set_trace()
# row = len(img)
# col = len(img[0])
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
# def afterfirst_ending_point(i, j, aa):
#     fix_row = row - 1
#     # import pdb; pdb.set_trace()
#     for j in range(j - 1, 0, -1):
#         for i in range(fix_row, 0, -1):
#             if int(img[i][j]) == 0:
#                 if (j + 1) in aa:
#                     return i, j
#                     # import pdb; pdb.set_trace()
#                 # ...
#                 # import pdb; pdb.set_trace()
#                 print("this is start point")
#     return afterfirst_ending_point()
#
# def start_point(i, j, point):
#     fix_row = row - 1
#     # import pdb; pdb.set_trace()
#     print("this is j check point .................! line 59", i, j )
#     for j in range(j-1, 0, -1):
#         for i in range(fix_row, 0, -1):
#             if int(img[i][j]) == 0:
#                 if (j-1) in point:
#                     print("this is j check point .................! line 64", i, j)
#
#                     import pdb;pdb.set_trace()
#                     return i, j
#                 # import pdb; pdb.set_trace()
#                 # ...
#                 # import pdb; pdb.set_trace()
#                 print("this is start point")
#     return start_point()
# def end_point(rows, col):
#     fix_row = row - 1
#     check_point = False
#     point = []
#     white_pixel_check =[]
#     # import pdb; pdb.set_trace()
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
#         # import pdb; pdb.set_trace()
#
#     return point
#
# row_cut_start_point = find_first_black_row(row, col)
# row_cut_end_point = find_last_black_row(row, col)
# # import pdb;pdb.set_trace()
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
#             if flage:
#                 cv2.rectangle(img, (start_point1[1], row_cut_start_point), (j, row-1), (30, 250, 12), 1)
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
#                 cv2.rectangle(img, (start_point1[1], row_cut_start_point), (ending_point[1], row), (30, 250, 12),1)
#                 img = Image.open('green-font-bbox.png')
#                 crop_img = img.crop((start_point1[1] + 1, row_cut_start_point, ending_point[1] + 1, row_cut_end_point))
#                 crop_img.save(f"last_image{str(crop_name)}.jpg")
#                 crop_img.show()
#                 img = np.array(img)
#                 after_first_box_ending_point.clear()
#                 print(after_first_box_ending_point, "after_first_box_ending_point")
#                 after_first_box_ending_point.append(start_point1[1] - 1)
#                 start_point1.clear()
#                 ending_point.clear()



































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
#                 # import pdb; pdb.set_trace()
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


import cv2
import numpy as np
import tensorflow as tf



import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
#
# # Load the MNIST dataset of digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Normalize the pixel values between 0 and 1
# x_train = x_train/255.0
# x_test = x_test/255.0
#
# # Reshape the images to include a channel dimension
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
#
# # Define the CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# # Compile the model with the appropriate loss function and optimizer
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
#
# # Save the model as an h5 file
# model.save('digit_recognition_model.h5')













image = cv2.imread("green-font.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
model = tf.keras.models.load_model('digit_recognition_model.h5')

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    # Check aspect ratio and size to determine if it is a 7-segment digit
    if aspect_ratio >= 0.2 and aspect_ratio <= 2.0 and w > 8 and h > 8:
        # Draw a bounding box around the digit
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the digit region of interest and resize it to the size required by the model
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = np.array(roi, dtype=np.float32)
        roi = roi.reshape(1, 28, 28, 1) / 255.0

        # Predict the digit using the pre-trained model
        pred = model.predict(roi)
        digit = np.argmax(pred)

        # Draw the predicted digit on the image
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)