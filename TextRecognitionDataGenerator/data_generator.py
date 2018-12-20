import os
import random

import albumentations as A
import cv2
import imgaug as ia
import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import (rank, threshold_niblack, threshold_sauvola)
from skimage.morphology import disk

import pyblur

from background_generator import BackgroundGenerator
from computer_text_generator import ComputerTextGenerator
from distorsion_generator import DistorsionGenerator
from elastic_distortion_generator import ElasticDistortionGenerator
from handwritten_text_generator import HandwrittenTextGenerator


def decision(probability):
    return random.uniform(0, 1) < probability


def sauvola_bin(img, thres=0.3):
    img = np.array(img)
    bin = img > threshold_sauvola(img, window_size=15, k=thres)
    img = bin.astype('uint8') * 255
    return img


def add_random_space_to_string(s):
    s = list(s)
    for i in range(len(s) - 1):
        if s[i] == ' ':
            while random.randrange(3):
                s[i] = s[i] + ' '
    return ''.join(s)


def nick_binarize(img_list):
    '''Binarize linecut images using two differently sized local threshold kernels

    Args:
        img_list: list of grayscale linecut images
    Returns:
        results: binarized images in the same order as the input'''

    results = []

    for img in img_list:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height = img.shape[0]
        width = img.shape[1]

        # Resize the images to 200 pixel height
        scaling_factor = 100/img.shape[0]
        new_w = int(scaling_factor*img.shape[1])
        new_h = int(scaling_factor*img.shape[0])
        # img = cv2.resize(img, (new_w, new_h))
        img = np.array(Image.fromarray(img).resize(
            (new_w, new_h), Image.ANTIALIAS))

        # First pass thresholding
        th1 = threshold_niblack(img, 13, 0.00)

        # Second pass thresholding
        radius = 101
        structured_elem = disk(radius)
        th2 = rank.otsu(img, structured_elem)

        # Masking
        img = (img > th1) | (img > th2)
        img = img.astype('uint8')*255

        img = np.array(Image.fromarray(img).resize(
            (width, height), Image.ANTIALIAS))
        results.append(img)

    return results


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(cls, index, text, font, out_dir, height, extension, skewing_angle, random_skew, blur, random_blur, background_type, distorsion_type, distorsion_orientation, is_handwritten, name_format, text_color=-1, prefix="", random_crop=False, debug=False):
        try:
            max_height = 80.0

            albu = A.Compose([
                A.RandomBrightness(limit=.1, p=0.3),
                A.RandomContrast(limit=.1, p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.CLAHE(p=0.3),
                A.HueSaturationValue(hue_shift_limit=20,
                                        sat_shift_limit=30,
                                        val_shift_limit=20, p=0.3),
                # A.ChannelShuffle(p=0.3),
                A.JpegCompression(quality_lower=95, p=0.3),
            ], p=1)

            #####################################
            # Generate name for resulting image #
            #####################################
            if name_format == 0:
                image_name = '{}_{}.{}'.format(text, str(index), extension)
            elif name_format == 1:
                image_name = '{}_{}.{}'.format(str(index), text, extension)
            elif name_format == 2:
                image_name = '{}.{}'.format(str(index), extension)
            elif name_format == 3:
                image_name = '{}_{}.{}'.format(prefix, str(index),
                                                extension)
            else:
                print(
                    '{} is not a valid name format. Using default.'.format(
                        name_format))
                image_name = '{}_{}.{}'.format(text, str(index), extension)
            # print(image_name, font)

            img = None

            ##########################
            # Create picture of text #
            ##########################
            add_random_space = ' ' in text and decision(0.02)

            text = "  " + text + "  "
            if (len(text) < 40):
                for x in range(random.randint(1, 3)):
                    text = " " + text

                for x in range(random.randint(1, 3)):
                    text = text + " "

            if add_random_space:
                text = add_random_space_to_string(text)

            text_mode = np.random.choice(
                5, 1, p=[0.86, 0.02, 0.02, 0.0, 0.1])[0]
            extend_bottom = np.random.choice(3, 1, p=[0.5, 0.3, 0.2])[0] + 2

            if is_handwritten:
                img = HandwrittenTextGenerator.generate(text)
            else:
                img = ComputerTextGenerator.generate(
                    text, font, text_color, height, text_mode=text_mode, extend_bottom=extend_bottom)

            img = np.array(img)
            img = img[random.randint(0, 2):, :]
            img = Image.fromarray(img)

            if debug:
                img.convert('L').save(
                    os.path.join(out_dir, image_name.replace(".jpg", "_7.jpg")))

            if decision(0.6):
                random_angle = random.uniform(-skewing_angle/4,
                                                skewing_angle/4)

                rotated_img = img.rotate(
                    skewing_angle if not random_skew else random_angle, expand=1)  # .resize(img.size)
            else:
                random_angle = random.uniform(-skewing_angle,
                                                skewing_angle)

                rotated_img = img.rotate(
                    skewing_angle if not random_skew else random_angle,
                    expand=1)

            # if decision(0.3):
            # rotated_img = Image.fromarray(scipy.ndimage.rotate(img, random_angle))
            # else:
            # rotated_img = Image.fromarray(imutils.rotate_bound(np.array(img), random_angle))
            #
            #     white_mask = Image.new('RGBA', rotated_img.size, (255,) * 4)
            #     rotated_img = Image.composite(rotated_img, white_mask, rotated_img)

            if debug:
                rotated_img.convert('L').save(
                    os.path.join(out_dir, image_name.replace(".jpg", "_6.jpg")))
            # rotated_img = rotated_img.convert('L')

            ###################################
            # Random miscellaneous distortion #
            ###################################

            if decision(0.7):
                rotated_img = cv2.cvtColor(
                    np.array(rotated_img), cv2.COLOR_GRAY2BGR)
                augmented = albu(image=rotated_img, mask=None, bboxes=[],)
                rotated_img = Image.fromarray(cv2.cvtColor(
                    augmented['image'], cv2.COLOR_BGR2GRAY))

            if decision(0.9):
                if decision(0.2):
                    if decision(0.5):
                        # full image erode
                        x = random.randint(0, 2)
                        kernel = np.ones((x, x), np.uint8)
                        im_arr = cv2.erode(
                            np.array(rotated_img), kernel, iterations=1)
                    else:
                        # partial image erode
                        im_arr = np.array(rotated_img)
                        start_x = random.randint(0, int(im_arr.shape[1] * 0.7))
                        if start_x + 10 < im_arr.shape[1]:
                            end_x = random.randint(
                                start_x + 10, im_arr.shape[1])
                            x = random.randint(1, 4)
                            kernel = np.ones((x, x), np.uint8)
                            im_arr[:, start_x:end_x] = cv2.erode(
                                im_arr[:, start_x:end_x], kernel, iterations=1)
                else:
                    im_arr = np.array(rotated_img)

                rotated_img = Image.fromarray(im_arr)

                if debug:
                    rotated_img.convert('L').save(
                        os.path.join(out_dir, image_name.replace(".jpg", "_5.jpg")))
                random_erode_pixel = decision(0.005)
                prob = 1.0
                if random_erode_pixel:
                    # random erode with random pixel sampling
                    x = random.randint(0, 2)
                    kernel = np.ones((x, x), np.uint8)
                    im_arr = np.array(rotated_img)
                    erode = cv2.erode(im_arr, kernel, iterations=1)
                    # prob = np.random.choice([0.1, 0.2, 0.3], p=[0.05, 0.3, 0.65])
                    prob = random.uniform(0.96, 1.0)
                    mask = np.random.choice(
                        2, im_arr.shape, p=[1 - prob, prob]).astype('uint8')
                    im_arr[mask > 0] = erode[mask > 0]
                    rotated_img = Image.fromarray(im_arr)
                    if debug:
                        rotated_img.convert('L').save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg", "_3.jpg")))

                random_pixel_discard = decision(0.2)

                if random_pixel_discard:
                    # random pixel discard
                    # print("lol")
                    im_arr = np.array(rotated_img)
                    # prob = np.random.choice([0.1, 0.15, 0.25], p=[0.6, 0.3, 0.1])
                    prob = random.uniform(0.95, 1.0)
                    mask = np.random.choice(
                        2, im_arr.shape, p=[1 - prob, prob]).astype('uint8')
                    im_arr[mask == 0] = 255
                    # im_arr = np.clip(im_arr, 0, 255).astype('uint8')
                    rotated_img = Image.fromarray(im_arr)

                    # seq = ia.augmenters.Sequential([ia.augmenters.Dropout(random.uniform(0,0.05))])
                    # rotated_img = Image.fromarray(seq.augment_image(np.array(rotated_img)))

                    if debug:
                        rotated_img.convert('L').save(
                            os.path.join(out_dir, image_name.replace(".jpg", "_4.jpg")))
                ######################################
                # Apply geometry distortion to image #
                ######################################

                distorsion_type = np.random.choice(
                    4, 1, p=[0.75, 0.15, 0.05, 0.05])[0]
                if distorsion_type == 0:
                    distorted_img = rotated_img  # Mind = blown
                elif distorsion_type == 1:
                    distorted_img = DistorsionGenerator.sin(
                        rotated_img,
                        vertical=(distorsion_orientation ==
                                    0 or distorsion_orientation == 2),
                        horizontal=(distorsion_orientation ==
                                    1 or distorsion_orientation == 2),
                        max_offset=2
                    )
                elif distorsion_type == 2:
                    distorted_img = DistorsionGenerator.cos(
                        rotated_img,
                        vertical=(distorsion_orientation ==
                                    0 or distorsion_orientation == 2),
                        horizontal=(distorsion_orientation ==
                                    1 or distorsion_orientation == 2),
                        max_offset=2
                    )
                elif not random_pixel_discard and distorsion_type == 3:
                    distorted_img = DistorsionGenerator.random(
                        rotated_img,
                        vertical=(distorsion_orientation ==
                                    0 or distorsion_orientation == 2),
                        horizontal=(distorsion_orientation ==
                                    1 or distorsion_orientation == 2)
                    )
                else:
                    distorted_img = DistorsionGenerator.cos(
                        rotated_img,
                        vertical=(
                            distorsion_orientation == 0 or distorsion_orientation == 2),
                        horizontal=(
                            distorsion_orientation == 1 or distorsion_orientation == 2),
                        max_offset=2
                    )
                new_text_width, new_text_height = distorted_img.size

                if debug:
                    distorted_img.convert('L').save(
                        os.path.join(out_dir, image_name.replace(".jpg", "_2.jpg")))

                affine_type = np.random.choice(4, 1, p=[0.1, 0.05, 0, 0.85])[0]
                if not random_pixel_discard or (random_pixel_discard is True and prob > 0.98):
                    if affine_type == 0 and distorted_img.size[1] > 40 and distorsion_type == 0:
                        distorted_img = ElasticDistortionGenerator.afffine_transform(
                            distorted_img)
                        if debug:
                            distorted_img.convert('L').save(os.path.join(out_dir,
                                                                            image_name.replace(
                                                                                ".jpg",
                                                                                "_1_1.jpg")))
                    elif affine_type == 1:
                        distorted_img = ElasticDistortionGenerator.elastic_transform(
                            distorted_img)

                        if debug:
                            distorted_img.convert('L').save(os.path.join(out_dir,
                                                                            image_name.replace(
                                                                                ".jpg",
                                                                                "_1_2.jpg")))
                    # elif affine_type == 2:
                    #     distorted_img = ElasticDistortionGenerator.perspective_transform(distorted_img)
                    #     distorted_img.convert('L').save(os.path.join(out_dir,
                    #                                                  image_name.replace(
                    #                                                      ".jpg",
                    #                                                      "_1_3.jpg")))

                if np.min(np.array(distorted_img)) > 250:
                    print(index, "2 wtf. why!!!",
                            affine_type, random_pixel_discard)

                x = random.randint(-3, 3)
                y = random.randint(1, 3)

                if debug:
                    distorted_img.convert('L').save(os.path.join(
                        out_dir, image_name.replace(".jpg", "_1.jpg")))
                #############################
                # Generate background image #
                #############################
                background_type = np.random.choice(
                    4, 1, p=[0.1, 0.3, 0.02, 0.58])[0]

                if background_type == 0:
                    background = BackgroundGenerator.gaussian_noise(
                        new_text_height + x, new_text_width + y)
                elif background_type == 1:
                    background = BackgroundGenerator.plain_white(
                        new_text_height + x, new_text_width + y)
                elif background_type == 2 and random_erode_pixel is False and random_pixel_discard is False:
                    background = BackgroundGenerator.quasicrystal(
                        new_text_height + x, new_text_width + y)
                elif random_erode_pixel is False and random_pixel_discard is False and distorsion_type != 3:
                    background = BackgroundGenerator.picture(
                        new_text_height + x, new_text_width + y)
                else:
                    background = BackgroundGenerator.gaussian_noise(
                        new_text_height + x, new_text_width + y)

                distorted_img = distorted_img.convert('L')
                mask = distorted_img.point(
                    lambda x: 0 if x == 255 or x == 0 else 255, '1')

                apply_background = False
                if (random.randint(0, 10) < 4):
                    background = distorted_img
                else:
                    apply_background = True
                    background.paste(distorted_img, (5, 5), mask=mask)

                ##################################
                # Resize image to desired format #
                ##################################
                # new_width = float(new_text_width + y) * \
                #     (float(height) / float(new_text_height + x))
                # image_on_background = background.resize((int(new_width), height), Image.ANTIALIAS)

                # if distorsion_type != 3 and background_type != 2 and new_text_height > 45:
                #     final_image = background.filter(
                #         ImageFilter.GaussianBlur(
                #             radius=(blur if not random_blur else random.randint(0, blur))
                #         )
                #     )
                # else:

                ##################################
                # Random motion blur             #
                ##################################
                final_image = background.convert('L')

                if debug:
                    final_image.save(
                        os.path.join(out_dir, image_name.replace(".jpg", "_0.jpg")))

                # final_image = Image.fromarray(nick_binarize([np.array(final_image)])[0])

                # random binary if background is white
                # if blur_type in [1, 2] and background_type in [0, 1] and decision(0.6) and distorsion_type != 3:
                #     bin_thres = 0.3 if blur_type == 2 else 0.03
                #     binary_im = Image.fromarray(sauvola_bin(final_image, thres=bin_thres))
                #     if np.mean(binary_im) > 160:
                #         final_image = binary_im
            else:
                final_image = rotated_img.convert("L")
                mask = final_image.point(
                    lambda x: 0 if x == 255 or x == 0 else 255, '1')
            
                new_text_width, new_text_height = final_image.size
                x = random.randint(-3, 3)
                y = random.randint(1, 3)
                background = BackgroundGenerator.plain_white(
                    new_text_height + x, new_text_width + y)
                apply_background = False
                background.paste(final_image, (5, 5), mask=mask)
                final_image = background.convert('L')
            resize_type = random.choice(
                [Image.ANTIALIAS, Image.BILINEAR, Image.LANCZOS])

            if decision(0.7):
                if (decision(0.5)):
                    f = random.uniform(
                        0.7, min(1.4, max_height/final_image.size[1]))
                    final_image = final_image.resize((int(
                        final_image.size[0] * f), int(final_image.size[1] * f)),
                        resize_type)

                    if decision(0.05):
                        f = 64.0/final_image.size[1]
                        final_image = final_image.resize((int(
                            final_image.size[0] * f), int(final_image.size[1] * f)),
                            resize_type)
                else:
                    if (random.randint(0, 1) == 0):
                        f = random.uniform(
                            0.6, min(1.2, max_height/final_image.size[1]))

                        final_image = final_image.resize(
                            (int(final_image.size[0] * f), int(final_image.size[1])), resize_type)
                    else:
                        f = random.uniform(
                            0.85, min(1.1, max_height/final_image.size[1]))

                        final_image = final_image.resize(
                            (int(final_image.size[0]), int(final_image.size[1] * f)), resize_type)

            # blur distortion
            blur_type = np.random.choice(
                5, 1, p=[0.15, 0.2, 0.25, 0.2, 0.2])[0]

            if decision(0.8) and distorsion_type != 2:
                if blur_type == 0:
                    final_image = pyblur.LinearMotionBlur_random(final_image)
                    if debug:
                        final_image.save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg",
                                                            "_0_0.jpg")))
                elif blur_type == 1:
                    final_image = pyblur.GaussianBlur_random(final_image)
                    if debug:
                        final_image.save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg",
                                                            "_0_1.jpg")))
                elif blur_type == 2:
                    kernel = np.ones((5, 5), np.float32) / \
                        random.randint(30, 50)
                    final_image = Image.fromarray(
                        cv2.filter2D(np.array(final_image), -1,
                                        kernel))
                    if debug:
                        final_image.save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg",
                                                            "_0_2.jpg")))
                elif blur_type == 3:
                    final_image = Image.fromarray(
                        cv2.blur(np.array(final_image), (5, 5)))

                    if debug:
                        final_image.save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg",
                                                            "_0_3.jpg")))
                elif blur_type == 4 and final_image.size[0] > 40 and apply_background is not True:
                    final_image = pyblur.PsfBlur_random(final_image)

                    if debug:
                        final_image.save(
                            os.path.join(out_dir,
                                            image_name.replace(".jpg",
                                                            "_0_4.jpg")))

            # additional sharpening
            if decision(0.1) and blur_type != 4:
                final_image = final_image.filter(ImageFilter.EDGE_ENHANCE)
                if debug:
                    final_image.save(
                        os.path.join(out_dir,
                                        image_name.replace(".jpg",
                                                        "_0_2.jpg")))

            seq = ia.augmenters.Sequential(ia.augmenters.OneOf([
                ia.augmenters.Affine(
                    shear=(-36, 36),
                    order=[0, 1],
                    cval=0,
                    mode=ia.ALL),
            ]))

            final_image = Image.fromarray(
                seq.augment_image(np.array(final_image)))
            
            # random invert
            inverted = False
            if blur_type != 4:
                if decision(0.3):
                    if (
                            background_type == 3 | distorsion_type | blur_type != 0):
                        if (decision(0.1)):
                            im_arr = np.array(final_image)
                            im_arr = np.bitwise_not(im_arr)
                            final_image = Image.fromarray(im_arr)
                            inverted = True
                    else:
                        im_arr = np.array(final_image)
                        im_arr = np.bitwise_not(im_arr)
                        final_image = Image.fromarray(im_arr)
                        inverted = True

                if decision(0.1):
                    if inverted:
                        seq = ia.augmenters.Sequential(
                            [ia.augmenters.Salt(random.uniform(0.02, 0.05))])
                        final_image = Image.fromarray(
                            seq.augment_image(np.array(final_image)))
                    else:
                        seq = ia.augmenters.Sequential(
                            [ia.augmenters.Pepper(random.uniform(0.02, 0.05))])
                        final_image = Image.fromarray(
                            seq.augment_image(np.array(final_image)))

            # Random crop
            if random_crop and decision(0.1):
                final_image = np.array(final_image)
                final_image = final_image[random.randint(10,20):,:]
                final_image = Image.fromarray(final_image)

            # Save the image
            final_image.convert('L').save(os.path.join(out_dir, image_name))
        except Exception as ex:
            print(ex)
