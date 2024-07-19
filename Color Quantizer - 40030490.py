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
    -----------------------------
    Attention: 
    Processing may take longer than 1 minute.
"""
import re
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Global variables --------------------------------------------------------------------------
pixels = []
img_rgb = []
ratio_list = []
quantized_images = []
k_list = [2, 4, 8] # some examples for different k values 

# Search for the pattern in the input (path) string -----------------------------------------
def get_image_name():
    match = re.search(pattern, path)
    return match.group(1)

# Step 1: Load and Preprocess the Image -----------------------------------------------------
def get_2d_image():
    global pixels, img_rgb
    
    # a) Read an image using OpenCV
    img_bgr = cv.imread(path)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)  # Convert image from BGR to RGB

    # b) Convert the image to a 2D array of pixels
    pixels = img_rgb.reshape((-1, 3))

    # c) Retrieve the dimensions of the image (height, width)
    return img_rgb.shape  # in 2D, third dimension is << don't care >> :)

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
def get_k():
    input_k = int(input("\nPlease enter the number of colors (final k): ")) # final k value as user input
    k_list.append(input_k)
    k_list.sort()


# Step 4: Create the Quantized Image --------------------------------------------------------
def image_quantizer(path, width, height, img_name):
    original_size = os.path.getsize(path) # original image size

    for k in k_list:
        centers, labels = kmeans(pixels, k)
        
        # a) Replace each pixel in the original image with the nearest centroid color from the K-Means result
        quantized_img = centers[labels.flatten().astype(int)].reshape((height, width, 3)).astype(np.uint8)

        # b) Reshape the 2D pixel array back into the original image shape
        quantized_img_bgr = cv.cvtColor(quantized_img, cv.COLOR_RGB2BGR)

        # c) Save the quantized image (for the current k-value) for comparison 
        if not os.path.exists("./Quantized images"):
            os.makedirs("./Quantized images")
        cv.imwrite(f"./Quantized images/{img_name}Colors-{k}.png", quantized_img_bgr)
        
        # Append quantized image in quantized_images list
        quantized_images.append(quantized_img)
        
        # Evaluate the result of current k
        evaluate_results(k, original_size)

# Step 5: Evaluate the Results --------------------------------------------------------------
def evaluate_results(k, original_size):
    quantized_size = os.path.getsize(f"./Quantized images/{image_name}Colors-{k}.png")
    compression_ratio = original_size / quantized_size # compression ratio of the current quantized image
    ratio_list.append(compression_ratio)

    print(f"\nWith k = {k}:")
    print(f"Original Image Size: {original_size} bytes")
    print(f"Quantized Image Size: {quantized_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")

# Step 6: Plot the Results ------------------------------------------------------------------
def visual_comparison():
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

def compression_comparison():
    plt.figure()
    plt.plot(k_list, ratio_list, marker='o')
    plt.xlabel('Number of colors (k)')
    plt.ylabel('Ratio (Original / Quantized) size')
    plt.title('Comparison of the compression ratio\nfor different k values')
    plt.grid(True)
    plt.show()

# Main part with functions call -------------------------------------------------------------
if __name__ == '__main__':
    os.system('cls')
    
    # Image path and necessary pattern 
    path = "./Main images/MyMaster.png"
    pattern = r"\/([A-Za-z0-9]+)\.png$"
    image_name = get_image_name()
    
    # Load and Preprocess the Image
    height, width, _ = get_2d_image()
    
    # Quaztizing images 
    get_k()
    image_quantizer(path, width, height, image_name)
    
    # Comparison of original and quantized image
    visual_comparison()
    compression_comparison()