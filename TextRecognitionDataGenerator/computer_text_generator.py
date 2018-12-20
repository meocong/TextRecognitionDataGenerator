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
RANDOM_ADDITIONAL_SPACE = 4

class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color, height, text_mode, extend_bottom=0):
        # print(text, font, text_color)
        # image_font = ImageFont.truetype(font="/Library/Fonts/Arial Unicode.ttf", size=32)

        start_pixel = 0
        N = len(text) // 2
        first_half = text[:N]
        second_half = text[N:]

        image_font = ImageFont.truetype(font=font, size=int(height))

        if (text_mode == RANDOM_BIG_TEXT):
            ## random letter font increase
            image_font_big = ImageFont.truetype(font=font, size=int(
                height * random.uniform(1.0, 1.1)))
        else:
            image_font_big = ImageFont.truetype(font=font, size=int(height * random.uniform(1.0, 1.5)))

        if text_mode == TIGHT_TEXT:
            ## draw letter by letter for tight text generation
            f = random.uniform(0.93, 0.96)
            text_width, text_height = image_font.getsize(text)
            txt_img = Image.new('L', (int(text_width * (f + 0.01)), text_height + extend_bottom), 255)
            txt_draw = ImageDraw.Draw(txt_img)

            offset_x = 0
            for c in text:
                char_w, char_h = image_font.getsize(c)
                txt_draw.text((offset_x, start_pixel), u'{0}'.format(c),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                offset_x += int(f * char_w)

        elif text_mode == SECOND_HALF_BIG_TEXT:
            # text_width, text_height = image_font.getsize(first_half)

            if (random.randint(0,1) == 0):
                temp = image_font
                image_font = image_font_big
                image_font_big = temp

            if (random.randint(0,2) != 0 or len(text) < 6):
                text_width1, text_height1 = image_font.getsize(first_half)
                text_width2, text_height2 = image_font_big.getsize(second_half)
                txt_img = Image.new('L', (text_width1 + text_width2, max(text_height1, text_height2) + extend_bottom), 255)
                txt_draw = ImageDraw.Draw(txt_img)

                txt_draw.text((0, start_pixel), u'{0}'.format(first_half), fill=random.randint(1, 50) if text_color < 0 else text_color,
                              font=image_font)
                txt_draw.text((image_font.getsize(first_half)[0], start_pixel), u'{0}'.format(second_half),
                              fill=random.randint(1, 50) if text_color < 0 else text_color,
                              font=image_font_big)
            else:
                N1 = random.randint(1,len(text) - 3)
                N2 = random.randint(N1 + 1, len(text))

                text_width1, text_height1 = image_font_big.getsize(text[:N1])
                text_width2, text_height2 = image_font.getsize(text[N1:N2])
                text_width3, text_height3 = image_font_big.getsize(text[N2:])
                txt_img = Image.new('L', (
                text_width1 + text_width2 + text_width3, max(max(text_height1, text_height2), text_height3 + extend_bottom)), 255)
                txt_draw = ImageDraw.Draw(txt_img)

                txt_draw.text((0, start_pixel), u'{0}'.format(text[:N1]),
                              fill=random.randint(1,
                                                  50) if text_color < 0 else text_color,
                              font=image_font_big)
                txt_draw.text((image_font_big.getsize(text[:N1])[0], start_pixel), u'{0}'.format(text[N1:N2]),
                              fill=random.randint(1,
                                                  50) if text_color < 0 else text_color,
                              font=image_font)
                txt_draw.text((image_font_big.getsize(text[:N1])[0] + image_font.getsize(text[N1:N2])[0], start_pixel), u'{0}'.format(text[N2:]),
                              fill=random.randint(1,
                                                  50) if text_color < 0 else text_color,
                              font=image_font_big)

        elif text_mode == RANDOM_BIG_TEXT:
            text_width, text_height = image_font_big.getsize(text)
            txt_img = Image.new('L', (text_width, text_height + extend_bottom), 255)
            txt_draw = ImageDraw.Draw(txt_img)

            offset_x = 0
            is_bigger = np.random.choice([0,1], (len(text),), p=[0.7, 0.3])
            for i, c in enumerate(text):
                font = image_font_big if is_bigger[i] else image_font
                char_w, char_h = font.getsize(c)
                txt_draw.text((offset_x, start_pixel), u'{0}'.format(c),
                              fill=random.randint(1, 50) if text_color < 0 else text_color,
                              font=font)
                offset_x += char_w

        elif text_mode == NORMAL_TEXT:
            text_width, text_height = image_font.getsize(text)
            txt_img = Image.new('L', (text_width, text_height + extend_bottom), 255)
            txt_draw = ImageDraw.Draw(txt_img)

            ## normal text print
            txt_draw.text((0, start_pixel), u'{0}'.format(text), fill=random.randint(1, 50) if text_color < 0 else text_color,
                          font=image_font)
        elif text_mode == RANDOM_ADDITIONAL_SPACE:
            ## draw letter by letter for additional random space
            text_width, text_height = image_font.getsize(text)

            addi = []
            for c in text:
                addi.append(np.random.choice(4, 1, p=[0.5, 0.2, 0.2, 0.1])[0])

            txt_img = Image.new('L', (text_width + np.sum(addi), text_height + extend_bottom), 255)
            txt_draw = ImageDraw.Draw(txt_img)

            offset_x = 0
            for i, c in enumerate(text):
                char_w, char_h = image_font.getsize(c)
                txt_draw.text((offset_x, start_pixel), u'{0}'.format(c),
                              fill=random.randint(1, 80) if text_color < 0 else text_color,
                              font=image_font)
                offset_x += char_w + addi[i]


        return txt_img
