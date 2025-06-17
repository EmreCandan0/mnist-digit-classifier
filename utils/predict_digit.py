import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("No digit found in image.")
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped = img[y:y+h, x:x+w]


    pil_img = Image.fromarray(cropped)
    pil_img = ImageOps.invert(pil_img)
    pil_img = ImageOps.pad(pil_img, (28, 28), color=255, centering=(0.5, 0.5))


    img_array = np.array(pil_img) / 255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)
    return img_array

def predict(path_to_img):
    model = load_model("mnist_cnn_model.h5")
    img_array = preprocess_image(path_to_img)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    print(f"Model prediction: {predicted_digit}")
    plt.imshow(img_array[0], cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict("test_inputs/my_digit.png")

