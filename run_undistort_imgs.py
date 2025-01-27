from undistort_toolbox import undistort_images_toolbox

config_dir = "/path/to/config"
image_dir = "/path/to/input/images"
output_dir = "/path/to/output/directory"

undistort_images_toolbox(
    config_dir=config_dir,
    image_dir=image_dir,
    output_dir=output_dir,
    file_extension="bmp",       # Supported formats: bmp, png, jpg, etc.
    crop_images=True,           # Crop based on ROI
    visualize=True,             # Show side-by-side comparison
    save_undistorted=True,      # Save undistorted images
    max_images=10               # Set to None to process all images
)
