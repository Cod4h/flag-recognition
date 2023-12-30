import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the saved model
loaded_model = tf.keras.models.load_model('flag_recognition_model_update2.h5')

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image = image.convert("RGB")  # Convert to RGB if necessary
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to handle the image selection
def select_image():
    global canvas

    # Allow user to select an image file
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )

    if file_path:
        try:
            # Preprocess the image
            image = preprocess_image(file_path)

            # Make predictions using the loaded model
            predictions = loaded_model.predict(image)[0]
            classes = ['argentinean', 'uruguayan']  # Add the class names

            # Calculate prediction percentages
            percentages = [round(pred * 100, 2) for pred in predictions]

            # Get the predicted class index and percentage
            class_index = np.argmax(predictions)
            class_percentage = percentages[class_index]

            # Display the selected image and predicted class with percentage
            display_image(file_path)
            result_label.configure(text=f"Flag: {classes[class_index]}, Percentage: {class_percentage}%")

            # Create a pie chart for class probabilities
            if canvas is not None:
                canvas.get_tk_widget().pack_forget()

            fig, ax = plt.subplots()
            
            # Define colors for the pie chart slices
            colors = ['#87CEEB', '#0038a8ff']
            
            ax.pie(percentages, labels=classes, autopct='%1.1f%%', colors=colors)
            ax.set_title("Class Probabilities")

            # Convert the figure to a Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()

            # Place the canvas in the UI
            canvas.get_tk_widget().pack()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Define a function to display the image in the GUI
def display_image(image_path):
    image = Image.open(image_path).resize((300, 300))
    image_tk = ImageTk.PhotoImage(image)
    image_label.configure(image=image_tk)
    image_label.image = image_tk

# Create the main window
window = tk.Tk()
window.title("Flag Recognition")
window.geometry("800x800")

# Create global variables for the canvas and image label
canvas = None
image_label = None

# Create a button to select an image
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(window)
image_label.pack()

# Create a label to display the predicted class
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()