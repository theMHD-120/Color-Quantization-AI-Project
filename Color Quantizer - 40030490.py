"""
    ||| In the name of ALLAH |||
    -----------------------------
    Seyed Mahdi Mahdavi Mortazavi 
    StdNo.: 40030490
    -----------------------------
    Artificial Intelligence (AI)
    Assignment: Final Project #02
    >- Image Processing:
    >>> Color Quantization
"""
import re
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')

# Image path and necessary pattern ----------------------------------------------------------
path = "./Main images/MyForza.png"
pattern = r"\/([A-Za-z0-9]+)\.(png|jpg)$"

# Search for the pattern in the input (path) string -----------------------------------------
match = re.search(pattern, path)
image_name = match.group(1)

# Step 1: Load and Preprocess the Image -----------------------------------------------------
# a) Read an image using OpenCV
img_bgr = cv.imread(path)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)  # Convert image from BGR to RGB

# b) Convert the image to a 2D array of pixels
pixels = img_rgb.reshape((-1, 3))

# c) Retrieve the dimensions of the image (height, width)
height, width, _ = img_rgb.shape  # in 2D, third dimension is << don't care >> :)

# Step 2: Implement K-Means Clustering ------------------------------------------------------
def kmeans(pixels, k, max_iters=100):
    # Initialize centroids
    np.random.seed(42)
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Assign each pixel to the nearest centroid
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([pixels[labels == j].mean(axis=0) for j in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        # Update if no convergence (equal or less than a fixed number of iterations (max_iters))
        centroids = new_centroids
        
    return centroids, labels

# Step 3: Comparison of original and quantized images ---------------------------------------
k_list = [2, 4, 8] # some examples for different k values 
input_k = int(input("\nPlease enter the number of colors: ")) # final k value as user input
k_list.append(input_k)
k_list.sort()

# initialization for comparisons
ratio_list = []
quantized_images = []
original_size = os.path.getsize(path) # original image size

for k in k_list:
    centers, labels = kmeans(pixels, k)
    
    # Step 4: Create the Quantized Image --------------------------------------------------------
    # a) Replace each pixel in the original image with the nearest centroid color from the K-Means result
    quantized_img = centers[labels.flatten().astype(int)].reshape((height, width, 3)).astype(np.uint8)

    # b) Reshape the 2D pixel array back into the original image shape
    quantized_img_bgr = cv.cvtColor(quantized_img, cv.COLOR_RGB2BGR)

    # c) Save the quantized image (for the current k-value) for comparison 
    if not os.path.exists("./Quantized images"):
        os.makedirs("./Quantized images")
    cv.imwrite(f"./Quantized images/{image_name}Colors-{k}.png", quantized_img_bgr)

    # Step 5: Evaluate the Results --------------------------------------------------------------
    quantized_images.append(quantized_img)
    quantized_size = os.path.getsize(f"./Quantized images/{image_name}Colors-{k}.png")
    compression_ratio = original_size / quantized_size # compression ratio of the current quantized image
    ratio_list.append(compression_ratio)

    print(f"\nWith k = {k}:")
    print(f"Original Image Size: {original_size} bytes")
    print(f"Quantized Image Size: {quantized_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")

# Step 6: Plot the Results ------------------------------------------------------------------
# a) Visual comparison
fig, axs = plt.subplots(1, len(k_list) + 1, figsize=(20, 5))

# Original image
axs[0].imshow(img_rgb)
axs[0].set_title("Original")
axs[0].axis('off')

# Quantized images
for i, k in enumerate(k_list):
    axs[i + 1].imshow(quantized_images[i])
    axs[i + 1].set_title(f"k = {k}")
    axs[i + 1].axis('off')

plt.tight_layout()
plt.show()

# b) Comparison of compression ratio
plt.figure()
plt.plot(k_list, ratio_list, marker='o')
plt.xlabel('Number of colors (k)')
plt.ylabel('Ratio (Original / Quantized) size')
plt.title('Comparison of the compression ratio\nfor different k values')
plt.grid(True)
plt.show()