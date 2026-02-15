import ctypes
import numpy as np
import logging
from rapidshot.processor.base import ProcessorBackends

# Set up logger
logger = logging.getLogger(__name__)


class NumpyProcessor:
    """
    NumPy-based processor for image processing.
    """

    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.NUMPY

    def __init__(self, color_mode):
        """
        Initialize the processor.

        Args:
            color_mode: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        """
        self.cvtcolor = None
        self.color_mode = color_mode
        self.PBYTE = ctypes.POINTER(ctypes.c_ubyte)

        # Simplified processing for BGRA
        if self.color_mode == "BGRA":
            self.color_mode = None

    def _get_pointer_address(self, ptr):
        """
        Get the integer address from a ctypes pointer.

        Args:
            ptr: A ctypes pointer object (c_void_p, POINTER, or int)

        Returns:
            int: The memory address

        Raises:
            ValueError: If the pointer is invalid
        """
        if isinstance(ptr, int):
            return ptr
        elif isinstance(ptr, ctypes.c_void_p):
            return ptr.value
        elif hasattr(ptr, "contents"):
            return ctypes.addressof(ptr.contents)
        else:
            raise ValueError("Invalid pointer")

    def process_cvtcolor(self, image):
        """
        Convert color format with robust error handling.

        Args:
            image: Image to convert

        Returns:
            Converted image
        """
        # Fixed region handling patch applied
        # Skip color conversion if image is None or empty
        if image is None or image.size == 0:
            logger.warning("Received empty image for color conversion")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Ensure image has proper shape and type
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except Exception as e:
                logger.warning(f"Failed to convert image to numpy array: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)

        # Handle images with no channels or wrong number of channels
        if len(image.shape) < 3 or image.shape[2] < 3:
            try:
                import cv2

                # Convert grayscale to BGR if needed
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return image
            except Exception as e:
                logger.warning(f"Failed to convert image format: {e}")
                return np.zeros(
                    (
                        image.shape[0] if len(image.shape) > 0 else 480,
                        image.shape[1] if len(image.shape) > 1 else 640,
                        3,
                    ),
                    dtype=np.uint8,
                )

        try:
            # Initialize color conversion function once, if not already done
            if self.cvtcolor is None:
                if self.color_mode == "RGB":
                    # BGRA to RGB: Select channels B, G, R and reverse them to R, G, B
                    self.cvtcolor = lambda img: img[..., [2, 1, 0]]
                elif self.color_mode == "BGR":
                    # BGRA to BGR: Select first three channels (B, G, R)
                    self.cvtcolor = lambda img: img[..., :3]
                elif self.color_mode == "RGBA":
                    # BGRA to RGBA: Make a copy (input is BGRA, effectively selecting all channels)
                    # OpenCV's BGRA2RGBA also just copies if the alpha is to be preserved.
                    self.cvtcolor = lambda img: img.copy()
                else:
                    # Fallback to OpenCV for other modes like GRAY or if color_mode is unexpected
                    try:
                        import cv2

                        color_mapping = {
                            # "RGB": cv2.COLOR_BGRA2RGB, # Handled by NumPy
                            # "RGBA": cv2.COLOR_BGRA2RGBA, # Handled by NumPy
                            # "BGR": cv2.COLOR_BGRA2BGR, # Handled by NumPy
                            "GRAY": cv2.COLOR_BGRA2GRAY
                            # Add other specific OpenCV conversions here if needed
                        }

                        if self.color_mode in color_mapping:
                            cv2_code = color_mapping[self.color_mode]
                            if cv2_code == cv2.COLOR_BGRA2GRAY:
                                # Add axis for grayscale to maintain shape consistency
                                self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)[
                                    ..., np.newaxis
                                ]
                            else:
                                self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)
                        else:
                            logger.warning(
                                f"Unsupported color mode: {self.color_mode} with NumPy. "
                                "Falling back to OpenCV BGR conversion."
                            )
                            # Default to BGR via OpenCV if mode is unknown and not handled by NumPy
                            self.cvtcolor = lambda img: cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    except ImportError:
                        logger.error(
                            "OpenCV is not installed, but required for color mode: {}".format(
                                self.color_mode
                            )
                        )
                        # Set a lambda that raises an error or returns image unchanged
                        # if cv2 is required but not found
                        self.cvtcolor = lambda img: img  # Or raise error
                    except Exception as cv_err:
                        logger.error(
                            f"Error initializing OpenCV converter for {self.color_mode}: {cv_err}"
                        )
                        self.cvtcolor = lambda img: img  # Fallback

            # Perform the conversion
            return self.cvtcolor(image)

        except Exception as e:
            logger.warning(f"Color conversion error for mode '{self.color_mode}': {e}")
            # Fallback: return BGR from BGRA if possible, or original image
            if image.ndim == 3 and image.shape[2] == 4:  # BGRA
                return image[..., :3]  # Return BGR part
            elif image.ndim == 3 and image.shape[2] == 3:  # Already 3 channels
                return image
            # If it's grayscale or some other format, return as is or a placeholder
            return image  # Or np.zeros(...) as per previous logic for severe errors

    def shot(self, image_ptr, rect, width, height):
        """
        Process directly to a provided memory buffer.

        Args:
            image_ptr: Pointer to image buffer
            rect: Mapped rectangle
            width: Width
            height: Height
        """
        try:
            pitch = int(rect.Pitch)
            row_bytes = width * 4

            # Get source and destination addresses
            src_address = self._get_pointer_address(rect.pBits)
            dst_address = self._get_pointer_address(image_ptr)

            if pitch == row_bytes:
                ctypes.memmove(dst_address, src_address, row_bytes * height)
            else:
                for row in range(height):
                    ctypes.memmove(
                        dst_address + row * row_bytes,
                        src_address + row * pitch,
                        row_bytes,
                    )
        except Exception as e:
            logger.error(f"Memory copy error: {e}")

    def process(self, rect, width, height, region, rotation_angle, output_buffer=None):
        """
        Process a frame using zero-copy numpy views.

        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle,
            output_buffer: Pre-allocated NumPy array to store the processed frame.
        """
        try:
            if not hasattr(rect, "pBits") or not rect.pBits:
                raise ValueError(f"Invalid rect or pBits, cannot process. Rect type: {type(rect)}")

            pitch = int(rect.Pitch)

            # Validate pitch is aligned to 4-byte boundaries (BGRA pixels)
            if pitch % 4 != 0:
                raise ValueError(f"Pitch {pitch} is not divisible by 4")

            # Get the pointer address
            src_address = self._get_pointer_address(rect.pBits)

            region_left, region_top, region_right, region_bottom = region
            if not (0 <= region_left < region_right <= width) or not (
                0 <= region_top < region_bottom <= height
            ):
                raise ValueError(
                    f"Region {region} is outside of the frame dimensions {(width, height)}"
                )

            region_height = region_bottom - region_top
            region_width = region_right - region_left

            # Create zero-copy numpy array directly over mapped memory
            # Key insight from BetterCam: use pitch in the shape to handle stride
            pitch_in_pixels = pitch // 4
            total_size = pitch * height

            # Validate buffer size matches expected dimensions
            expected_size = height * pitch_in_pixels * 4
            if total_size != expected_size:
                raise ValueError(
                    f"Buffer size mismatch: total_size={total_size}, "
                    f"expected={expected_size}"
                )

            src_buffer = (ctypes.c_ubyte * total_size).from_address(src_address)

            # Create view over entire mapped memory with pitch as width
            image = np.ctypeslib.as_array(src_buffer).reshape((height, pitch_in_pixels, 4))

            # Extract the region using numpy slicing (zero-copy)
            # This handles both the region extraction and pitch correction in one step
            image = image[region_top:region_bottom, region_left:region_right, :]

            # Now handle output buffer logic
            if output_buffer is None:
                # No pool: copy data to avoid returning view into mapped memory
                # The caller unmaps immediately after process() returns,
                # so a view would point to unmapped memory
                is_pooled_buffer = False
                current_array = np.copy(image)
            else:
                is_pooled_buffer = True
                if (
                    output_buffer.shape[:2] != (region_height, region_width)
                    or output_buffer.shape[2] != 4
                ):
                    raise ValueError(
                        f"Output buffer shape {output_buffer.shape} does not match region shape "
                        f"({region_height}, {region_width}, 4)."
                    )
                # Copy data into the pooled buffer
                output_buffer[:] = image
                current_array = output_buffer

            is_still_pooled_buffer = is_pooled_buffer

            # Color Conversion
            if self.color_mode is not None:
                converted_array = self.process_cvtcolor(current_array)

                if (
                    converted_array.shape[0] == current_array.shape[0]
                    and converted_array.shape[1] == current_array.shape[1]
                ):
                    # If number of channels changed (e.g. to BGR or GRAY)
                    if converted_array.shape[2] != current_array.shape[2]:
                        current_array = converted_array
                        is_still_pooled_buffer = False
                    elif (
                        converted_array.base is not current_array.base
                        and converted_array is not current_array
                    ):
                        # It's a copy with the same shape
                        if is_still_pooled_buffer:
                            current_array[:] = converted_array
                else:
                    logger.warning("Color conversion changed height/width, which is unexpected.")
                    current_array = converted_array
                    is_still_pooled_buffer = False

            # Rotation
            if rotation_angle != 0:
                k = (rotation_angle // 90) % 4
                if k != 0:
                    rotated_array = np.rot90(current_array, k=k)

                    # Check if shape changed due to rotation
                    if (
                        rotated_array.shape[0] != current_array.shape[0]
                        or rotated_array.shape[1] != current_array.shape[1]
                    ):
                        current_array = rotated_array
                        is_still_pooled_buffer = False
                    elif is_still_pooled_buffer:
                        current_array[:] = rotated_array
                    else:
                        current_array = rotated_array

            return current_array, is_still_pooled_buffer

        except Exception as e:
            logger.error(f"Frame processing error in NumpyProcessor: {e}")
            # Ensure output_buffer is zeroed out in case of any error,
            # then return it with False flag
            if output_buffer is not None and hasattr(output_buffer, "fill"):
                try:
                    output_buffer.fill(0)
                except Exception as fill_e:
                    logger.error(f"Error filling output_buffer after another error: {fill_e}")
            return output_buffer, False
