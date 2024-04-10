# ImageCropper
Imagecropper is a tool to enable efficient cropping and adjusting of multiple scanned or photographed images.  Control is through keyboard shortcuts.  Usage is as follows:
```
python imagecropper.py input_image_file output_image_file [--initial-key-presses bgn...]
```
or
```
python imagecropper.py input_image_directory output_image_directory [--initial-key-presses bgn...]
```

In the first case, the input image file is loaded, and presented in an interactive window.  The initial key presses are applied as if they had been typed
at the keyboard.  If the user chooses to save an image, it will be written to output_image_file, or if output_image_file exists, it will be written to a file
with the same name but suffixed by (1), (2), etc

In the second case, all image files in all subdirectories are found.  If a corresponding output is found at the same tree position in the output directory, it is skipped.  If no corresponding output is found the image is edited as above, and when saved is saved to the corresponding file in the output directory tree.  Again, if multiple extracts are saved, they will be saved to successive files 
```
output_image_name (1).suffix
output_image_name (2).suffix
output_image_name (3).suffix
...
```

The available commands can be viewed on screen by pressing '?'