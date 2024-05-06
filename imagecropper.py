import cv2
import argparse
import os
import re
import numpy as np
from pathlib import Path
from tkinter import Tk

BRIGHTNESS_INCREMENT = 10
CONTRAST_INCREMENT = 0.1
ROTATION_INCREMENT = 0.5


def equalizeHist(image):
    image = image.copy()
    for c in range(3):
        image[:, :, c] = cv2.equalizeHist(image[:, :, c])
    return image
    # # convert from RGB color-space to YCrCb
    # ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # # equalize the histogram of the Y channel
    # ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # # convert back to RGB color-space from YCrCb
    # return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


class ImageCropper:
    def __init__(self, image):
        self.image = image
        self.reset_color()
        self.reset_geometry()
        self.get_screen_info()
        self.speed_factor = 1
        self.dirty = True
        self.help_text = "\n".join(
            [
                "? - show/hide this help",
                "s - save and continue with this image",
                "q - save and move to next image",
                "x - skip this image",
                "z - exit now without saving",
                "+/- - increase/decrease speed of changes",
                "g/b - increase/decrease brightness",
                "d/c - increase/decrease contrast",
                "t/r - increase/decrease rotation",
                "w/u - flip left-right / up-down",
                "e/y - rotate 90 left / right",
                "8/i - adjust top boundary",
                "h/j - adjust left boundary",
                "k/l - adjust right boundary",
                "m/, - adjust bottom boundary",
                "! - reset color",
                "@ - reset geometry",
                "1/2 - decrease/increase red",
                "3/4 - decrease/increase green",
                "5/6 - decrease/increase blue",
                "n - invert",
                "~ - toggle grayscale",
                "= - toggle equalize histogram",
            ]
        )
        self.show_help = False

    def reset_color(self):
        self.brightness = 0
        self.contrast = 1
        self.invert = False
        self.grayscale = False
        self.equalize_histogram = False
        self.color_factors = np.array((1, 1, 1))
        self.dirty = True

    def reset_geometry(self):
        self.rotation = 0
        self.rot90 = 0
        self.crop_window = (0, 0, self.image.shape[1], self.image.shape[0])
        self.dirty = True

    def get_screen_info(self):
        window = Tk()
        self.screen_width = window.winfo_screenwidth()
        self.screen_height = window.winfo_screenheight()
        window.destroy()

    def flip_lr(self):
        self.image = self.image[:, ::-1, :]
        x_low, y_low, x_high, y_high = self.crop_window
        self.crop_window = (
            self.image.shape[1] - x_high,
            y_low,
            self.image.shape[1] - x_low,
            y_high,
        )
        self.dirty = True

    def flip_ud(self):
        self.image = self.image[::-1, :, :]
        x_low, y_low, x_high, y_high = self.crop_window
        self.crop_window = (
            x_low,
            self.image.shape[0] - y_high,
            x_high,
            self.image.shape[0] - y_low,
        )
        self.dirty = True

    def rot90left(self):
        x_low, y_low, x_high, y_high = self.crop_window
        self.crop_window = (
            y_low,
            self.image.shape[1] - x_high,
            y_high,
            self.image.shape[1] - x_low,
        )
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.dirty = True


    def rot90right(self):
        x_low, y_low, x_high, y_high = self.crop_window
        self.crop_window = (
            self.image.shape[0] - y_high,
            x_low,
            self.image.shape[0] - y_low,
            x_high,
        )
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.dirty = True

    def increment_crop_window(self, side, increment):
        x_low, y_low, x_high, y_high = self.crop_window
        if side == "left":
            x_low = max(0, min(x_low + increment, x_high - 1))
        elif side == "right":
            x_high = min(self.image.shape[1], max(x_high + increment, x_low + 1))
        elif side == "top":
            y_low = max(0, min(y_low + increment, y_high - 1))
        elif side == "bottom":
            y_high = min(self.image.shape[0], max(y_high + increment, y_low + 1))
        self.crop_window = (x_low, y_low, x_high, y_high)

    def set_crop_window(self, crop_window):
        self.crop_window = crop_window

    def set_brightness(self, brightness):
        self.brightness = brightness
        self.dirty = True

    def set_contrast(self, contrast):
        self.contrast = contrast
        self.dirty = True

    def set_rotation(self, rotation):
        self.rotation = rotation
        self.dirty = True

    def set_color_factors(self, color_factors):
        self.color_factors = color_factors
        self.dirty = True

    def toggle_invert(self):
        self.invert = not self.invert
        self.dirty = True

    def toggle_grayscale(self):
        self.grayscale = not self.grayscale
        self.dirty = True

    def toggle_equalize_histogram(self):
        self.equalize_histogram = not self.equalize_histogram
        self.dirty = True

    def generate_image(self):
        if not self.dirty:
            return self.adjusted_image
        image = self.image.copy()

        # Rotate the image
        if self.rotation != 0:
            (h, w) = image.shape[:2]
            if self.crop_window is None:
                center = (w // 2, h // 2)
            else:
                x_low, y_low, x_high, y_high = self.crop_window
                center = ((x_low + x_high) // 2, (y_low + y_high) // 2)
            M = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        # else:
        #     image = self.image
        if self.invert:
            image = cv2.bitwise_not(image)

        # Apply brightness and contrast adjustment
        image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=self.brightness)

        # Apply color adjustment
        image = np.clip(image * self.color_factors, 0, 255).astype(np.uint8)

        # Convert to grayscale if necessary
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.equalize_histogram:
            if self.crop_window is None:
                image = equalizeHist(image)
            else:
                x_low, y_low, x_high, y_high = self.crop_window
                image[y_low:y_high, x_low:x_high] = equalizeHist(
                    image[y_low:y_high, x_low:x_high]
                )
        self.adjusted_image = image
        self.dirty = False
        return image

    def generate_save_image(self):
        image = self.generate_image()
        # Apply crop window
        if self.crop_window is not None:
            x_low, y_low, x_high, y_high = self.crop_window
            image = image[y_low:y_high, x_low:x_high]

        return image

    def generate_interactive_image(self):
        image = self.generate_image()
        # Determine the scale ratio to fit the image on the screen
        scale_ratio = min(
            (self.screen_width - 10) / image.shape[1],
            (self.screen_height - 100) / image.shape[0],
        )

        # Resize the image if necessary
        if scale_ratio < 1:
            image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)

        # Apply crop window
        if self.crop_window is not None:
            x_low, y_low, x_high, y_high = self.crop_window
            cv2.rectangle(
                image,
                (int(x_low * scale_ratio), int(y_low * scale_ratio)),
                (int((x_high) * scale_ratio), int((y_high) * scale_ratio)),
                (0, 255, 0),
                2,
            )

        # Display the help text
        if self.show_help:
            y = 20
            for line in self.help_text.split("\n"):
                cv2.putText(
                    image,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 20

        return image

    def process_key_press(self, key):
        if key == 255:
            return
        cropstep = max(1, int(16 * self.speed_factor))
        match chr(key):
            case "/" | "?":  # Toggle help text
                self.show_help = not self.show_help
            case "+":  # Press '+' to increase speed of changes
                self.speed_factor = min(self.speed_factor * 2, 16)
            case "-":  # Press '-' to decrease speed of changes
                self.speed_factor = max(self.speed_factor / 2, 1 / 16)
            case "c":  # Press 'c' to decrease contrast
                self.set_contrast(
                    self.contrast - CONTRAST_INCREMENT * self.speed_factor
                )
            case "d":  # Press 'd' or 'D' to increase contrast
                self.set_contrast(
                    self.contrast + CONTRAST_INCREMENT * self.speed_factor
                )
            case "n":  # Press 'n' to invert
                self.toggle_invert()
            case "b":  # Press 'b' to decrease brightness
                self.set_brightness(
                    self.brightness - BRIGHTNESS_INCREMENT * self.speed_factor
                )
            case "g":  # Press 'g' to increase brightness
                self.set_brightness(
                    self.brightness + BRIGHTNESS_INCREMENT * self.speed_factor
                )
            case "r":  # Press 'r' to decrease rotation
                self.set_rotation(
                    self.rotation - ROTATION_INCREMENT * self.speed_factor
                )
            case "t":  # Press 'R' to increase rotation
                self.set_rotation(
                    self.rotation + ROTATION_INCREMENT * self.speed_factor
                )
            case "w":  # Press 'f' to flip lr
                self.flip_lr()
            case "u":  # Press 'F' to flip up
                self.flip_ud()
            case "e":  # Press 'g' to rot90 left
                self.rot90left()
            case "y":  # Press 'G' to rot90 right
                self.rot90right()
            case "8":  # Press 'i' to adjust top boundary
                self.increment_crop_window("top", -cropstep)
            case "i":  # Press 'i' to adjust top boundary
                self.increment_crop_window("top", +cropstep)
            case "h":  # Press 'j' to adjust left boundary
                self.increment_crop_window("left", -cropstep)
            case "j":  # Press 'j' to adjust left boundary
                self.increment_crop_window("left", +cropstep)
            case "k":  # Press 'k' to adjust right boundary
                self.increment_crop_window("right", -cropstep)
            case "l":  # Press 'k' to adjust right boundary
                self.increment_crop_window("right", +cropstep)
            case "m":  # Press 'm' to adjust bottom boundary
                self.increment_crop_window("bottom", +cropstep)
            case ",":  # Press 'm' to adjust bottom boundary
                self.increment_crop_window("bottom", -cropstep)
            case "!":  # reset color
                self.reset_color()
            case "@":  # reset geometry
                self.reset_geometry()
            case "1":  # decrease red
                self.set_color_factors(
                    self.color_factors - np.array((0, 0, 0.05 * self.speed_factor))
                )
            case "2":  # increase red
                self.set_color_factors(
                    self.color_factors + np.array((0, 0, 0.05 * self.speed_factor))
                )
            case "3":  # decrease green
                self.set_color_factors(
                    self.color_factors - np.array((0, 0.05 * self.speed_factor, 0))
                )
            case "4":  # increase green
                self.set_color_factors(
                    self.color_factors + np.array((0, 0.05 * self.speed_factor, 0))
                )
            case "5":  # decrease blue
                self.set_color_factors(
                    self.color_factors - np.array((0.05 * self.speed_factor, 0, 0))
                )
            case "6":  # increase blue
                self.set_color_factors(
                    self.color_factors + np.array((0.05 * self.speed_factor, 0, 0))
                )
            case "=":
                self.toggle_equalize_histogram()
            case "~":
                self.toggle_grayscale()
            case _:
                print(f"Unknown key press: {chr(key)}")


def process_image(input_image, initial_key_presses, output_image):

    output_image = Path(output_image)
    # Load the image
    image = cv2.imread(input_image)
    imagecropper = ImageCropper(image)

    # Display the image and wait for key presses
    # Wait for commands
    position_in_key_presses = 0
    in_initial_key_presses = True
    while True:
        if initial_key_presses and position_in_key_presses < len(initial_key_presses):
            key = ord(initial_key_presses[position_in_key_presses])
            position_in_key_presses += 1
        else:
            in_initial_key_presses = False
            key = cv2.waitKey(1) & 0xFF
        match chr(key):
            # Press 'q' to quit and save, 's' to save and keep working, 'x' to skip
            case "q" | "s":
                # Save the image to a new file
                # check if output image exists
                while output_image.exists():
                    if x := re.match(r"(.*)\((\d+)\)(\..*)", output_image.name):
                        start, middle, end = x.groups()
                        output_image = output_image.parent / (
                            start + f"({int(middle)+1})" + end
                        )
                    else:
                        output_image = output_image.parent / (
                            output_image.stem + " (1)" + output_image.suffix
                        )
                print(f"Saving image to {output_image}")
                if not output_image.parent.exists():
                    output_image.parent.mkdir(parents=True)
                cv2.imwrite(str(output_image), imagecropper.generate_save_image())
                if key == ord("q"):
                    break
            case "x":
                break
            case "z":
                return False
            case _:
                imagecropper.process_key_press(key)
                if not in_initial_key_presses:
                    cv2.imshow("Image", imagecropper.generate_interactive_image())
    return True


def main(input_path, initial_key_presses, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        print("Input and output paths must be different.")
        return
    # Check if input path is an image
    if input_path.is_file():
        # Check if output path is an image or a directory
        if cv2.haveImageWriter(str(output_path)):
            # Call process_image with the input and output paths
            process_image(input_path, initial_key_presses, output_path)
        else:
            process_image(
                input_path, initial_key_presses, output_path / input_path.name
            )
    # Check if input path is a directory
    elif input_path.is_dir():
        # Verify that output path is a directory or does not exist
        if not output_path.is_dir() and output_path.exists():
            print("Output path must be a directory or a non-existing path.")
            return
        # Walk the directory and all subdirectories
        flag = True
        for root, dirs, files in os.walk(str(input_path)):
            for file in files:
                input_image_path = os.path.join(root, file)
                # Check if the file is an image
                if not cv2.haveImageReader(input_image_path):
                    continue
                # Generate the equivalent output path
                output_image_path = os.path.join(
                    output_path, os.path.relpath(input_image_path, str(input_path))
                )
                # Skip if the output file already exists
                if Path(output_image_path).exists():
                    continue
                # Call process_image with the input and output paths
                flag = process_image(
                    input_image_path, initial_key_presses, output_image_path
                )
                if not flag:
                    break
            if not flag:
                break
    else:
        print("Input path must be an image or a directory.")

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Image Cropper")

    # Add the input image argument
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input image or directory of images.  Can descend trees",
    )

    # Add the input image argument
    parser.add_argument(
        "--initial-key-presses",
        type=str,
        help="Execute these key presses before starting the interactive session",
    )

    # Add the output image argument
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output image, if exists will exit, if directory will save all images there in same structure as input.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.input_path, args.initial_key_presses, args.output_path)
