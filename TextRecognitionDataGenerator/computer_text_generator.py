import cv2
import math
import random
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter
# from fontTools.ttLib import TTFont
# from fontTools.unicode import Unicode
from itertools import chain

class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color, height, random_height_inc = True, tight_text = False):
        # print(text, font, text_color)
        # image_font = ImageFont.truetype(font="/Library/Fonts/Arial Unicode.ttf", size=32)

        N = len(text) // 2
        first_half = text[:N]
        second_half = text[N:]

        image_font = ImageFont.truetype(font=font, size=height)
        image_font_big = ImageFont.truetype(font=font, size=int(height * 1.5))

        if random_height_inc:
            text_width_1, text_height_1 = image_font.getsize(first_half)
            text_width_2, text_height_2 = image_font_big.getsize(second_half)
            text_width = text_width_1 + text_width_2
            text_height = text_height_2

        else:
            text_width, text_height = image_font.getsize(text)

        # text = u'日産コーポレート/個人ゴールドJBC123JAL'
        txt_img = Image.new('L', (text_width, text_height), 255)

        txt_draw = ImageDraw.Draw(txt_img)

        if not tight_text:
            ## normal text print
            if random_height_inc:
                text_width, text_height = image_font.getsize(first_half)
                txt_draw.text((0, 0), u'{0}'.format(first_half), fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                txt_draw.text((text_width, 0), u'{0}'.format(second_half),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font_big)
            else:
                txt_draw.text((0, 0), u'{0}'.format(text), fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
        else:
            ## draw letter by letter for tight text generation
            offset_x = 0
            for c in text:
                char_w, char_h = image_font.getsize(c)
                txt_draw.text((offset_x, 0), u'{0}'.format(c), fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                offset_x += 0.9 * char_w

        return txt_img
