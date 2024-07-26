import cv2
import numpy as np


class Segmentix():
    def __init__(self):
        pass

    def resize_mask(
        self, ref_mask: np.ndarray, longest_side: int = 256
    ) :
        """
        Resize an image to have its longest side equal to the specified value.

        Args:
            ref_mask (np.ndarray): The image to be resized.
            longest_side (int, optional): The length of the longest side after resizing. Default is 256.

        Returns:
            tuple[np.ndarray, int, int]: The resized image and its new height and width.
        """
        height, width = ref_mask.shape[:2]
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))

        return (
            cv2.resize(
                ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            ),
            new_height,
            new_width,
        )

    def pad_mask(
        self,
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
    ) -> np.ndarray:
        """
        Add padding to an image to make it square.

        Args:
            ref_mask (np.ndarray): The image to be padded.
            new_height (int): The height of the image after resizing.
            new_width (int): The width of the image after resizing.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The padded image.
        """

        pad_height = 256 - new_height
        pad_width = 256 - new_width
        if pad_all_sides:
            padding = (
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            )
        else:
            padding = ((0, pad_height), (0, pad_width))

        # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
        return np.pad(ref_mask, padding, mode="constant")

    def reference_to_sam_mask(
        self, ref_mask: np.ndarray, pad_all_sides: bool = False
    ) -> np.ndarray:
        """
        Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.

        Args:
            ref_mask (np.ndarray): The grayscale mask to be processed.
            threshold (int, optional): The threshold value for the binarization. Default is 127.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The processed binary mask.
        """

        # Convert a grayscale mask to a binary mask.
        # Values over the threshold are set to 1, values below are set to -1.
        # ref_mask = np.clip((ref_mask > threshold) * 2 - 1, -1, 1)

        # Resize to have the longest side 256.
        resized_mask, new_height, new_width = self.resize_mask(ref_mask)

        # Add padding to make it square.
        square_mask = self.pad_mask(resized_mask, new_height, new_width, pad_all_sides)

        # Turn to logits mask input
        logits_mask = np.log((square_mask + 1e-16)/ (1-square_mask+1e-16))

        # Expand SAM mask's dimensions to 1xHxW (1x256x256).

        return np.expand_dims(logits_mask, axis=0)

    def turn_logits_to_possibility(self, logits, original_shape):
        # turn logits to 0-1
        logits = np.clip(logits, 36, -100)
        poss = np.exp(logits) / (np.exp(logits) + 1)
        height, width = original_shape

        # find the resize region scale
        longest_side = 256
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))

        # clip the region and resize to original size
        corres_region = poss[:new_height, : new_width][0]
        original_possibility = cv2.resize(corres_region, dsize = original_shape)
        return original_possibility