from PIL import Image
import easyocr
import numpy as np


def extract_text_from_image(image_file):
    reader = easyocr.Reader(['en'])
    extracted_text = ""
    try:
        image = Image.open(image_file)
        image_np = np.array(image)
    except IOError:
        return extracted_text
    text_list = reader.readtext(image_np, detail = 0)
    print(text_list)
    for i in text_list:
        extracted_text = extracted_text + " " + str(i)
    return extracted_text

print(extract_text_from_image("../test/t3.png"))