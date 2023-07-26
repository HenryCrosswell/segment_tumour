import os
import glob
import random 
import shutil

def make_val_dataset(training_folder, val_split):

    # get image files
    image_files = glob.glob(os.path.join(training_folder, '**/*.jpg'), recursive=True)

    selected_images = random.sample(image_files, int(len(image_files)*val_split))

    # Move the selected images to the validation directory
    for image_path in selected_images:
        # get the class name from the parent directory
        class_folder = os.path.basename(os.path.dirname(image_path))

        # create a new folder in the validation directory for the current class
        val_class_path = os.path.join('Validation', class_folder)
        if not os.path.exists(val_class_path):
            os.makedirs(val_class_path)

        # Move the image to the corresponding validation class folder
        val_image_path = os.path.join(val_class_path, os.path.basename(image_path))
        shutil.move(image_path, val_image_path)