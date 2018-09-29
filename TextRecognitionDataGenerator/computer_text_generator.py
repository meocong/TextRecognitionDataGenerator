import cv2
import math
import random
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter
# from fontTools.ttLib import TTFont
# from fontTools.unicode import Unicode
from itertools import chain

NORMAL_TEXT = 0
SECOND_HALF_BIG_TEXT = 1
TIGHT_TEXT = 2
RANDOM_BIG_TEXT = 3

class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color, height, text_mode):
        # print(text, font, text_color)
        # image_font = ImageFont.truetype(font="/Library/Fonts/Arial Unicode.ttf", size=32)

        N = len(text) // 2
        first_half = text[:N]
        second_half = text[N:]

        image_font = ImageFont.truetype(font=font, size=int(height))
        image_font_big = ImageFont.truetype(font=font, size=int(height * random.uniform(1.0, 1.5)))

        if text_mode in [SECOND_HALF_BIG_TEXT, RANDOM_BIG_TEXT]:  ## second half with bigger font
            text_width_1, text_height_1 = image_font.getsize(first_half)
            text_width_2, text_height_2 = image_font_big.getsize(second_half)
            text_width = text_width_1 + text_width_2 + 10
            text_height = max(text_height_1, text_height_2)
            if text_mode == RANDOM_BIG_TEXT:
                text_width = int(text_width * 1.1)
        else:
            text_width, text_height = image_font.getsize(text)

        # text = u'日産コーポレート/個人ゴールドJBC123JAL'
        txt_img = Image.new('L', (int(text_width*1.1), int(text_height*1.1) + 12), 255)

        txt_draw = ImageDraw.Draw(txt_img)

        if text_mode == TIGHT_TEXT:
            ## draw letter by letter for tight text generation
            offset_x = 0
            for c in text:
                char_w, char_h = image_font.getsize(c)
                txt_draw.text((offset_x, 6), u'{0}'.format(c),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                offset_x += 0.93 * char_w

        elif text_mode == SECOND_HALF_BIG_TEXT:
            text_width, text_height = image_font.getsize(first_half)

            if (random.randint(0,1) == 0):
                temp = image_font
                image_font = image_font_big
                image_font_big = temp

            if (random.randint(0,2) != 0 or len(text) < 6):
                txt_draw.text((0, 6), u'{0}'.format(first_half), fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                txt_draw.text((text_width, 6), u'{0}'.format(second_half),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font_big)
            else:
                N1 = random.randint(1,len(text) - 3)
                N2 = random.randint(N1 + 1, len(text))

                txt_draw.text((0, 6), u'{0}'.format(text[:N1]),
                              fill=random.randint(1,
                                                  80) if text_color < 0 else text_color,
                              font=image_font_big)
                txt_draw.text((image_font_big.getsize(text[:N1])[0], 6), u'{0}'.format(text[N1:N2]),
                              fill=random.randint(1,
                                                  80) if text_color < 0 else text_color,
                              font=image_font)
                txt_draw.text((image_font_big.getsize(text[:N1])[0] + image_font.getsize(text[N1:N2])[0], 6), u'{0}'.format(text[N2:]),
                              fill=random.randint(1,
                                                  80) if text_color < 0 else text_color,
                              font=image_font_big)

        elif text_mode == RANDOM_BIG_TEXT:
            ## random letter font increase
            image_font_big = ImageFont.truetype(font=font, size=int(
                height * random.uniform(1.0, 1.1)))

            offset_x = 0
            is_bigger = np.random.choice([0,1], (len(text),), p=[0.7, 0.3])
            for i, c in enumerate(text):
                font = image_font_big if is_bigger[i] else image_font
                char_w, char_h = font.getsize(c)
                txt_draw.text((offset_x, 6), u'{0}'.format(c),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=font)
                offset_x += char_w

        elif text_mode == NORMAL_TEXT:
            ## normal text print
            txt_draw.text((0, 6), u'{0}'.format(text), fill=random.randint(1, 80) if text_color < 0 else text_color,
                          font=image_font)


        return txt_img
