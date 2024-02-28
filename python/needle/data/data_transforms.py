import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        res = np.zeros(shape=[img.shape[0] + 2 * self.padding,
                                    img.shape[1] + 2 * self.padding,
                                    img.shape[2]])
        start_x = start_y = self.padding
        start_x += shift_x
        start_y += shift_y
        res[self.padding : self.padding + img.shape[0],
            self.padding : self.padding + img.shape[1],
            :] = img
        return res[start_x : start_x + img.shape[0],
                    start_y : start_y + img.shape[1],
                    :]
