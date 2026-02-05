import cv2


def rec_encoding_to_cv_color(encoding):
    """
    Get number of channels and cv2 color conversion for rec encoding.
    """

    # The encoding is equal to the pylon encoding of enum PixelFormatEnums
    if encoding == 32:  # bayerBG encoding
        channels = 1
        conv = cv2.COLOR_BayerBG2BGR
    elif encoding == 60:  # YCbCr422_8 encoding
        channels = 2
        conv = cv2.COLOR_YUV2BGR_YUY2
    elif encoding == 55:  # rgb
        channels = 3
        conv = cv2.COLOR_RGB2BGR
    else:  # legacy .rec files
        raise AttributeError(f'unknown encoding {encoding}')

    return channels, conv
