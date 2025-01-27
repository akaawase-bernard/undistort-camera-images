import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import matplotlib.pyplot as plt

def undistort_images_toolbox(config_dir, 
                             image_dir, 
                             output_dir, 
                             file_extension="bmp", 
                             crop_images=True, 
                             visualize=False, 
                             save_undistorted=True, 
                             max_images=None):
    """
    A versatile function for image undistortion with various flags and options.

    Parameters:
        config_dir (str): Path to the directory containing intrinsics_00.xml and distortion_00.xml.
        image_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where undistorted images will be saved.
        file_extension (str): File extension for input images (e.g., 'bmp', 'jpg', 'png').
        crop_images (bool): Whether to crop the undistorted image based on ROI.
        visualize (bool): Whether to display the original and undistorted images side by side.
        save_undistorted (bool): Whether to save the undistorted images to the output directory.
        max_images (int or None): Maximum number of images to process (None = process all images).
    """
    def load_intrinsics_and_distortion(config_dir):
        """Load intrinsics and distortion coefficients from XML files."""
        # Paths to the XML files
        intrinsics_file = os.path.join(config_dir, "intrinsics_00.xml")
        distortion_file = os.path.join(config_dir, "distortion_00.xml")

        # Load intrinsics
        tree = ET.parse(intrinsics_file)
        root = tree.getroot()
        intrinsics = [
            float(value) for value in root.find("intrinsics_penne/data").text.split()
        ]
        intrinsics = np.array(intrinsics, dtype=np.float32).reshape((3, 3))

        # Load distortion coefficients
        tree = ET.parse(distortion_file)
        root = tree.getroot()
        distortion = [
            float(value) for value in root.find("intrinsics_penne/data").text.split()
        ]
        distortion = np.array(distortion, dtype=np.float32).reshape((-1, 1))

        return intrinsics, distortion

    def plot_original_and_undistorted(original, undistorted, title1, title2):
        """Plot the original and undistorted images side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title(title1)
        axes[0].axis("off")

        # Undistorted image
        axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    # Load camera parameters
    intrinsics, distortion = load_intrinsics_and_distortion(config_dir)
    print("Camera intrinsics and distortion coefficients loaded.")

    # Create output directory if it doesn't exist
    if save_undistorted:
        os.makedirs(output_dir, exist_ok=True)

    # Get all image files with the specified extension
    image_files = glob(os.path.join(image_dir, f"*.{file_extension}"))
    if max_images:
        image_files = image_files[:max_images]

    for idx, image_path in enumerate(image_files, 1):
        print(f"Processing image {idx}/{len(image_files)}: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Get image size
        h, w = image.shape[:2]

        # Compute the optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (w, h), 1, (w, h)
        )

        # Undistort the image
        undistorted_image = cv2.undistort(image, intrinsics, distortion, None, new_camera_matrix)

        # Crop the image (optional, based on ROI)
        if crop_images:
            x, y, w, h = roi
            undistorted_image = undistorted_image[y:y+h, x:x+w]

        # Save the undistorted image
        if save_undistorted:
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, undistorted_image)
            print(f"Saved undistorted image to: {output_path}")

        # Visualize the images side by side
        if visualize:
            plot_original_and_undistorted(image, undistorted_image, "Original Image", "Undistorted Image")

    print("Image undistortion completed.")

# Example usage:
if __name__ == "__main__":
    config_dir = "/path/to/config"
    image_dir = "/path/to/input/images"
    output_dir = "/path/to/output/directory"

    undistort_images_toolbox(
        config_dir=config_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        file_extension="bmp",
        crop_images=True,
        visualize=True,
        save_undistorted=True,
        max_images=5  # Set to None to process all images
    )
