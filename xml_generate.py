import cv2

xml_data = """<annotation>
        <folder>font</folder>
        <filename>my_image_name</filename>
        <path>my_image_path</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>my_width</width>
            <height>my_height</height>
            <depth>1</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>my_digit</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>my_xmin</xmin>
                <ymin>my_ymin</ymin>
                <xmax>my_xmax</xmax>
                <ymax>my_ymax</ymax>
            </bndbox>
        </object>
    </annotation>"""

lst = ['/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0a72d48cfd97587c9440c615d67c7cac3c398eb7_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0a0216c77785d90a243e961676e1f4a95e1b1618_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0d2f85e9ce05fc8ae303cb62d5f12284605eebe9_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0d3fbca3c72473d8277fe990396127d22c6dc549_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0d9a0966ffe42c77d0c31bafb40095fc6f08ce50_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0e40a9dec5f22ddee0eeef87531212c9897daba3_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/1c3a1695baa8639a34118790f2490a2ff20e4b6d_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/1d335098977d3daf11051b28dea187de9e2c96d9_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/3c823ec6e7a7405211c7d9281432a406929eff97_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/05b575b71a82163a7bdb082904a7d553f32f66a6_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/07c660547cba0d1af70fd0ede8c4fb1046407dcf_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/12eaf64c705f59843fee2458d0f442246077beac_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/14d0c2db1882288556b7c4aa8676da86df501bc4_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/26f0a94f2cb7b8637e3f5339799d5f4ba1029024_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/34bd9ee3b020d9cd5297d6990784719bc68f2f2e_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/36a73c74ef0b76639e12488651f587fb06a9baab_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/184bad9043c572a63c3e2f838e79955faf946181_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/186a10c09ccc9fc9a3d1234f2e4d6c22869b21ed_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/1607ca8b8adc4399ba26923cf33db9b515cd8282_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/2321df655acee5f5edff8e504870401442eeb1b6_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/840502a0ad0ee55ef41b7a051cc86d9471559cb1_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/1093516de236b9fe08a1f68c66411177dd6d3914_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/22693110d9402616cddb934a401ebc21f18f3d57_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/0307110579f9699db48441ab71f3617acd055cdc_2.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/c1119af716de4f4954efb087092a49ee3b48be13_3.jpg','/home/plutusdev/Downloads/Seven-Segment-OCR-master/Datasets_digits/9/e47f55e29fb263bcdda378a8fc9fcc14cb532653_3.jpg']
for img in lst:
    name_with_ext = img.split('/')[-1]
    name = img.split('/')[-1].split('.')[0]

    # Read image
    im_in = cv2.imread(img)
    height = str(im_in.shape[0])
    width = str(im_in.shape[1])

    temp_xml_data = xml_data
    temp_xml_data = temp_xml_data.replace('my_image_name', name_with_ext)
    temp_xml_data = temp_xml_data.replace('my_image_path', img)
    temp_xml_data = temp_xml_data.replace('my_width', width)
    temp_xml_data = temp_xml_data.replace('my_height', height)
    temp_xml_data = temp_xml_data.replace('my_digit', 'nine')
    temp_xml_data = temp_xml_data.replace('my_xmin', '0')
    temp_xml_data = temp_xml_data.replace('my_ymin', '0')
    temp_xml_data = temp_xml_data.replace('my_xmax', width)
    temp_xml_data = temp_xml_data.replace('my_ymax', height)

    save_path_file = f"{name}.xml"
    with open(save_path_file, "w") as f:
        f.write(temp_xml_data)
    f.close()
