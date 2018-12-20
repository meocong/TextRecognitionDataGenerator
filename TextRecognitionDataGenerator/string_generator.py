import multiprocessing
import random
import re
import string
from itertools import chain

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os

def create_strings_from_file(filename, count, max_length):
    """
        Create all strings by reading lines in specified files
    """

    strings = []

    with open(filename, 'r', encoding="utf8") as f:
        lines = [l.strip()[0:max_length] for l in f.readlines()]
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        strings = random.sample(lines, count)

    return strings


def create_strings_from_dict(length, allow_variable, count, lang_dict):
    """
        Create all strings by picking X random word in the dictionary
    """

    dict_len = len(lang_dict)
    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            current_string += lang_dict[random.randrange(dict_len)][:-1]
            current_string += ' '
        strings.append(current_string[:-1])
    return strings


def decision(probability):
    return random.random() < probability


def random_insert_sym(word, pool):
    word = list(word)
    for i, _ in enumerate(word):
        if decision(0.33):
            while decision(0.33):
                word[i] += random.choice(pool)
    return ''.join(word)


def create_string_from_dict_with_random_chars(length, allow_variable, count, lang_dict, num=True, sym=False):
    """
        Create all strings by picking X random word in the dictionary
    """
    pool = ''
    if num:
        pool += "0123456789"
    if sym:
        pool += "!\"%'()+,-./:;\\_"

    dict_len = len(lang_dict)
    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            word = lang_dict[random.randrange(dict_len)][:-1]
            if decision(0.2):
                word = random_insert_sym(word, pool)
            current_string += word
            current_string += ' '
        strings.append(current_string[:-1])
    return strings


def query_wikipedia(args):
    lang, min_length, _ = args
    page = requests.get(
        'https://{}.wikipedia.org/wiki/Special:Random'.format(lang))

    soup = BeautifulSoup(page.text, 'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    # Only take a certain length
    lines = list(filter(
        lambda s:
        len(s.split(" ")) > min_length
        and not "Wikipedia" in s and not "This page was" in s
        and not "wikipedia" in s and not "Not logged in" in s and not "This article" in s
        and not "Jump to " in s and not "PDF" in s and not "Book" in s
        and not "Cookie" in s
        and re.match(r"our server", s.lower()) is None
        and re.match(r"ur server", s.lower()) is None
        and re.match(r"see the error", s.lower()) is None
        and re.match(r"if you report", s.lower()) is None
        and not s[0] == "^"
        and not "What links here" in s,
        [
            ' '.join(re.findall(r"[\w'@!\"#$%&()*+,\-./:;<=>?[\]^_`{|}~€¢³ðŸ¦±°‰¶§£¥·“”≪≫➡【】・くぐ〇〜ゝゞヽヾ一©®①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯]+",
                                s.strip()))[0:random.randint(1, 100)] for s in soup.get_text().splitlines()
        ]
    ))
    return lines[0:max([1, len(lines) - 5])]


def create_strings_from_wikipedia(minimum_length, count, lang, max_lines_per_page=3, nb_workers=8):
    """
        Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    sentences = []
    print("Generating strings from wiki...")
    pool = multiprocessing.Pool(nb_workers)
    files = [(lang, minimum_length, max_lines_per_page)] * \
        int(count / max_lines_per_page * 2)
    it = pool.imap_unordered(query_wikipedia, files)
    for lines in it:
        print(len(lines))
        sentences += lines

    return sentences[0:count]


def create_strings_from_fonts(fonts):
    strings = []
    font_dicts = {}
    for font in fonts:
        if (font not in font_dicts):
            ttf = TTFont(font, fontNumber=0)

            chars = [u'{0}'.format(chr(x[0])) for x in
                     list(chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables))]
            font_dicts[font] = chars
        else:
            chars = font_dicts[font]

        strings.append(''.join(random.choice(chars)
                               for i in range(random.randint(0, 100))))
    return strings


def check_character_in_font(char, font):
    try:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    return True
    except Exception as ex:
        print(ex)
        print(u'1{}1'.format(char))
    return False


def check_character_in_fontc1(char, font, height=32):
    image_font = ImageFont.truetype(font=font, size=height)
    text_width, text_height = image_font.getsize(char)

    # text_height = random.randint(text_height, text_height + 30)
    # text = u'日産コーポレート/個人ゴールドJBC123JAL'
    txt_img = Image.new('L', (text_width, text_height), 255)

    txt_draw = ImageDraw.Draw(txt_img)

    txt_draw.text((0, 0), u'{0}'.format(char), fill=0, font=image_font)

    txt_img = np.array(txt_img)
    # cv2.imshow("img", txt_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # txt_img = cv2.cvtColor(txt_img, cv2.COLOR_RGB2GRAY)
    return len(np.nonzero(np.array(txt_img) != 255)[0]) > 0


def check_character_in_fontc2(char, font, height=32):
    if (char in ["ロ"]):
        return True

    image_font = ImageFont.truetype(font=font, size=height)
    text_width, text_height = image_font.getsize(char)
    txt_img = Image.new('L', (text_width, text_height), 255)
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((0, 0), u'{0}'.format(char), fill=0, font=image_font)
    txt_img = np.array(txt_img)
    gray = txt_img
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]

    if (len(contours) < 4):
        for cnt in contours:
            approx = cv2.approxPolyDP(
                cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if (len(approx) == 4):
                (_, _, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                if ar >= 0.8 and ar <= 1.2:
                    return False
    else:
        pass

    return True


def random_latin(fonts):
    generated_list = []
    latin_chars = [x[:-1] for x in
                   open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [x[:-1] for x in open("dicts/special_char_latin_random.txt",
                                          encoding="utf-8").readlines()]

    max_len = 100
    for font in fonts:
        font = set(font)
        generated = ""

        latin_chars_in_font = [x for x in latin_chars if x in font]
        special_chars_in_font = [x for x in special_chars if x in font] + [
            " "] * 5

        for _ in range(3):
            for _ in range(random.randint(1, 10)):
                generated += random.choice(latin_chars_in_font)

            generated += random.choice(special_chars_in_font)

        if (len(generated) < max_len):
            for _ in range(random.randint(0, max_len - len(generated))):
                generated += random.choice(latin_chars_in_font)

        generated_list.append(generated)

    return generated_list


def random_latin_space(fonts):
    generated_list = []
    latin_chars = [x[:-1] for x in
                   open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [x[:-1] for x in open("dicts/special_char_latin.txt",
                                          encoding="utf-8").readlines()]

    # max_len = 100
    for font in fonts:
        font = set(font)
        generated = ""

        latin_chars_in_font = [x for x in latin_chars if x in font]
        special_chars_in_font = [x for x in special_chars if x in font]

        char_in_font = latin_chars_in_font + special_chars_in_font

        for _ in range(random.randint(2, 5)):
            for _ in range(random.randint(1, 10)):
                generated += random.choice(char_in_font)

            for _ in range(random.randint(2, 10)):
                generated += " "

        generated_list.append(generated)

    return generated_list


def gen_check_font(fonts):
    generated_list = []
    latin_chars = [x[:-1] for x in
                   open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [x[:-1] for x in open("dicts/special_char_latin.txt",
                                          encoding="utf-8").readlines()]

    # max_len = 100
    for font in fonts:
        font = set(font)

        latin_chars_in_font = [x for x in latin_chars if x in font]
        special_chars_in_font = [x for x in special_chars if x in font] + [
            " "] * 5

        generated_list.append(
            "".join(latin_chars_in_font + special_chars_in_font))

    return generated_list


def random_space(fonts):
    generated_list = []

    max_len = 75
    for _ in fonts:
        temp = ""
        for _ in range(random.randint(5, max_len)):
            temp += " "

        generated_list.append(temp)

    return generated_list


def generate_char_map_from_font(fonts, pre_font_dics={}):

    latin_chars = [x[:-1]
                   for x in open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [
        x[:-1] for x in open("dicts/special_char.txt", encoding="utf-8").readlines()]
    japan_chars = [x[:-1]
                   for x in open("dicts/japan.txt", encoding="utf-8").readlines()]
    # full_chars = latin_chars + special_chars + japan_chars
    font_dicts = {}
    # max_length = 60

    for font in tqdm(fonts):
        print("Generating font char maps...")
        print(font)
        if (font not in font_dicts and font not in pre_font_dics):
            ttf = TTFont(font, fontNumber=0)

            chars = set([u'{0}'.format(chr(x[0])) for x in
                         list(chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables))])
            japan_chars_in_font = [x for x in japan_chars if check_character_in_font(x, ttf)
                                   and check_character_in_fontc1(x, font)
                                   and check_character_in_fontc2(x, font)
                                   and x in chars]
            latin_chars_in_font = [x for x in latin_chars if check_character_in_font(x, ttf)
                                   and check_character_in_fontc1(x, font)
                                   and check_character_in_fontc2(x, font)
                                   and x in chars]
            special_chars_in_font = [x for x in special_chars if check_character_in_font(x, ttf)
                                     and check_character_in_fontc1(x, font)
                                     and check_character_in_fontc2(x, font)
                                     and x in chars] + [" " for x in range(1, 5)]

            font_dicts[font] = japan_chars_in_font + \
                latin_chars_in_font + special_chars_in_font
        elif font in font_dicts:
            pass
        elif font in pre_font_dics:
            font_dicts[font] = pre_font_dics[font]
        os.system('clear')
    return font_dicts


def random_sequences_sjnk(fonts):
    generated_list = []
    latin_chars = [x[:-1]
                   for x in open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [
        x[:-1] for x in open("dicts/special_char.txt", encoding="utf-8").readlines()]
    japan_chars = [x[:-1]
                   for x in open("dicts/japan.txt", encoding="utf-8").readlines()]

    for font in fonts:
        font = set(font)
        generated = ""

        latin_chars_in_font = [x for x in latin_chars if x in font]
        japan_chars_in_font = [x for x in japan_chars if x in font]
        special_chars_in_font = [
            x for x in special_chars if x in font] + [" "]*5

        for _ in range(3):
            for _ in range(random.randint(1, 15)):
                generated += random.choice(japan_chars_in_font)

            generated += random.choice(special_chars_in_font)
            generated += random.choice(latin_chars_in_font)
            if (random.randint(0, 10) == 0):
                generated += " "

        for _ in range(random.randint(0, 69-len(generated))):
            generated += random.choice(japan_chars_in_font)

        generated_list.append(generated)

    return generated_list


def create_strings_randomly(length, allow_variable, count, let, num, sym, lang):
    """
        Create all strings by randomly sampling from a pool of characters.
    """

    # If none specified, use all three
    if True not in (let, num, sym):
        let, num, sym = True, True, True

    pool = ''
    if let:
        if lang == 'cn':
            # Unicode range of CHK characters
            pool += ''.join([chr(i) for i in range(19968, 40908)])
        else:
            pool += string.ascii_letters
    if num:
        pool += "0123456789"
    if sym:
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

    if lang == 'cn':
        min_seq_len = 1
        max_seq_len = 2
    else:
        min_seq_len = 2
        max_seq_len = 10

    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            seq_len = random.randint(min_seq_len, max_seq_len)
            current_string += ''.join([random.choice(pool)
                                       for _ in range(seq_len)])
            current_string += ' '
        strings.append(current_string[:-1])
    return strings


def print_text(file, list_):
    f = open(file, 'w', encoding="utf-8")
    f.writelines(list_)


def gen_one_character(font_dicts: {}):
    fonts_arr = []
    strings = []

    for font in font_dicts.keys():
        for _ in range(0, 100):
            font_chars = font_dicts[font]
            generated = ""

            for _ in range(random.randint(1, 5)):
                generated += random.choice(font_chars)

            fonts_arr.append(font)
            strings.append(generated)

    return fonts_arr, strings
