import os
import sys
from PIL import Image, ImageDraw

input_image = sys.argv[1]
output_image = sys.argv[2]
h = int(sys.argv[3])
w = int(sys.argv[4])

def draw_circle(image_path, center, radius, output_path):
    # Open the image
    img = Image.open(image_path)
    print(img.size)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Calculate the bounding box of the circle
    x, y = center
    left_top = (x - radius, y - radius)
    right_bottom = (x + radius, y + radius)
    bounding_box = [left_top, right_bottom]

    # Draw the circle on the image
    draw.ellipse(bounding_box, outline="red", fill="red", width=2)

    # Save or display the result
    img.save(output_path)

center = (w, h)
radius = 20

draw_circle(input_image, center, radius, output_image)

