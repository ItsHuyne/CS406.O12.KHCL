import os.path

from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
IMG_SIZE = 512
import matplotlib.pyplot as plt
import main

def map(image, mask):
    img_np = np.array(image)

    # Expand dimensions of the tensor from (512, 512) to (512, 512, 1) to match the image's shape
    tensor = tf.expand_dims(mask, axis=-1)
    tensor = tf.image.resize(tensor, (img_np.shape[0], img_np.shape[1]), 'bilinear')

    # Convert numpy array to tensorflow tensor
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

    # Perform element-wise multiplication
    result = img_tensor * tensor

    # If you want the result as a PIL Image, convert it back
    result_img = Image.fromarray(np.uint8(result.numpy()))
    result_array = np.array(result_img)
    return result_array

# Định nghĩa hàm để lấy mặt nạ từ mô hình
def get_mask(model, file):
    # Chuyển đổi file từ streamlit UploadedFile sang file hình ảnh
    img = Image.open(file)
    img = img.convert('RGB') if img.mode == 'RGBA' else img
    img.save("img.png")

    image = cv2.imread("img.png", cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)

    # Dự đoán mặt nạ từ mô hình
    pred = model.predict(x, verbose=0)
    y0 = pred[0][0].reshape(512, 512)

    # Chuyển đổi y0 từ giá trị 0-1 thành 0-255 và lưu dưới dạng hình ảnh
    y0_image = (y0 * 255).astype(np.uint8)
    cv2.imwrite('mask.png', y0_image)
    cv2.imwrite('result.png', y0)

    return y0


# Hàm này sẽ được gọi từ main.py để xử lý file được tải lên và lưu mặt nạ
def process_and_save_mask(file):
    if file is not None:
        # Load mô hình đã được huấn luyện
        model = load_model('U2Net_AutoMattingData-0.6424-weights-10.h5', compile=False)
        # Lấy mặt nạ và lưu vào máy
        get_mask(model, file)
        # Xóa file img.png sau khi đã lấy mặt nạ để tránh lưu nhiều hình ảnh không cần thiết
        if os.path.exists('img.png'):
            os.remove('img.png')
