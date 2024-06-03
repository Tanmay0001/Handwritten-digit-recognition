from tkinter import *
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw

# Importing the necessary libraries for model and GUI
from keras.models import load_model

# Load the pre-trained model
model = load_model('archive (1).zip\t10k-labels-idx1-ubyte - ZIP64 archive, unpacked size 109,900,096 bytes') 
 # Replace 'mnist.h5' with the path to your trained model

def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert image to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Reshape to support model input and normalize
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # Predict the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

def clear_canvas():
    canvas.delete("all")
    label_status.config(text="Draw a digit and click Recognize")

def recognize_digit():
    # Get the coordinate of the canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    # Take a screenshot of the area drawn on the canvas
    image = ImageGrab.grab((x, y, x1, y1))
    # Predict the digit
    digit, acc = predict_digit(image)
    label_status.config(text=f"Predicted digit: {digit}, Confidence: {int(acc * 100)}%")

# Creating the GUI window
root = Tk()
root.title("Handwritten Digit Recognition")

# Creating Canvas to draw digits
canvas = Canvas(root, width=300, height=300, bg="white")
canvas.pack()

# Adding Buttons
btn_recognize = Button(root, text="Recognize", command=recognize_digit)
btn_recognize.pack(side=LEFT, padx=10)

btn_clear = Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(side=LEFT)

# Label to show the status
label_status = Label(root, text="Draw a digit and click Recognize", font=("Arial", 12))
label_status.pack(side=BOTTOM, pady=20)

# Function to handle drawing on Canvas
def draw(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black")

canvas.bind("<B1-Motion>", draw)

root.mainloop()