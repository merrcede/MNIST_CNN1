import io
import requests
from PIL import Image

# URL endpoint
url = 'http://127.0.0.1:5001/predict'

# Open the image and save it as a temporary file
image_path = r"C:\Users\AsusIran\Desktop\pic.png"
image = Image.open(image_path)

# Convert image to bytes to send
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format=image.format)
img_byte_arr = img_byte_arr.getvalue()

# Send a POST request to the API
response = requests.post(url, files={'image': img_byte_arr})


print(response.json())
