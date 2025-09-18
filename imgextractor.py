import google.generativeai as genai
import os
from PIL import Image
import numpy as np
import cv2


def extract_text_image(image_path):

    file_bytes=np.asarray(bytearray(image_path.read()),dtype=np.uint8)
    img=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR) 
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    image_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,image_bw=cv2.threshold(image_grey,90,255,cv2.THRESH_BINARY) # To convert to black and white

    # The image in CV2 gives is in numpy array.
    final_image=Image.fromarray(image_bw)

    # Configure Genai Model

    key=os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    model=genai.GenerativeModel("gemini-2.5-flash-lite")

    # Prompt

    prompt='''You act as an OCR application on the given image and extract the text from it.
    Give only the text as output and do not give any explanation or description.
    '''

    # Lets extract and return the text
    response=model.generate_content([prompt,final_image])
    output_text=response.text
    return output_text
