import cv2


def resize_img(model_image, input_image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 500.0 / model_image.shape[1]
    dim = (500, int(model_image.shape[0] * r))

    # perform the actual resizing of the image and show it
    model_image = cv2.resize(model_image, dim, interpolation = cv2.INTER_AREA)
    input_image = cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

    return (model_image, input_image)