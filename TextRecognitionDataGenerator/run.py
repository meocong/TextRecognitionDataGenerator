# -*- coding: utf-8 -*-
import argparse
import errno
import glob
import multiprocessing
import os
import random

from PIL import ImageFile
from tqdm import tqdm

from data_generator import FakeTextDataGenerator
from string_generator import (create_string_from_dict_with_random_chars,
                              create_strings_from_file,
                              create_strings_from_fonts,
                              create_strings_from_wikipedia,
                              create_strings_randomly, gen_check_font,
                              gen_one_character, generate_char_map_from_font,
                              print_text, random_latin, random_latin_space,
                              random_sequences_sjnk, random_space)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="./images/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-rsff",
        "--random_sequences_from_font",
        action="store_true",
        help="Use random sequences by characters in random fonts as the source text for the generation.",
        default=False
    )
    parser.add_argument(
        "-rdo",
        "--random_one_character",
        action="store_true",
        help="Use random single characters in random fonts as the source text for the generation.",
        default=False
    )
    parser.add_argument(
        "-sjnk",
        "--random_sequences_sjnk",
        action="store_true",
        help="Generate random sequences string from random font for sjnk project",
        default=False
    )
    parser.add_argument(
        "-sjnk_latin",
        "--random_latin_sjnk",
        action="store_true",
        help="Generate random latin string from random font for sjnk project",
        default=False
    )
    parser.add_argument(
        "-rdlt",
        "--random_latin_space",
        action="store_true",
        help="Generate random latiin string with random space from random font for sjnk project",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-rp",
        "--random_space",
        action="store_true",
        help="Generate random space images",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension (eg: .jpg, .png) to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-pre",
        "--prefix",
        type=str,
        nargs="?",
        help="Define the extension to save the image with, for example with prefix = abc, output name will be abc_1.jpg, abc_2.jpg",
        default="",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=float,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-ck",
        "--check_font",
        action="store_true",
        help="Unknown",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        action="store_true",
        help="Apply gaussian blur to the resulting sample or not",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-rc",
        "--random_crop",
        action="store_true",
        help="When set, generated images will be randomly crop at the top within 10-20 pixel range",
        default=False,
    )
    parser.add_argument(
        "-dg",
        "--debug",
        action="store_true",
        help="When set, multiple images will be generated at each step",
        default=False,
    )

    return parser.parse_args()


def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict


def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """
    # if lang == 'cn':
    return glob.glob('fonts/{0}/*.ttf'.format(lang)) + \
        glob.glob('fonts/{0}/*.ttc'.format(lang)) + \
        glob.glob('fonts/{0}/*.otf'.format(lang))
    # else:
    #     return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]


def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Create font (path) list
    fonts = load_fonts(args.language)
    # print(fonts)
    # Creating synthetic sentences (or word)
    strings = []

    if not args.check_font:
        fonts_arr = [fonts[random.randrange(0, len(fonts))]
                     for _ in range(0, args.count)]
    else:
        fonts_arr = fonts

    import pickle
    fonts_dict_path = './fonts/' + args.language + '/font_dict.pkl'
    try:
        fonts_dict = pickle.load(open(fonts_dict_path, "rb"))
        print("Loaded fonts dict from", fonts_dict_path)
        font_charsets = [fonts_dict[font] for font in fonts_arr]
    except:
        fonts_dict = {}
        fonts_dict = generate_char_map_from_font(fonts, fonts_dict)
        if not os.path.isdir('./fonts/' + args.language):
            os.system('mkdir ./fonts/' + args.language)
        pickle.dump(fonts_dict, open(fonts_dict_path, "wb"))
        print("Saved fonts dict to", fonts_dict_path)
        font_charsets = [fonts_dict[font] for font in fonts_arr]

    # print(fonts_dict)
    if args.use_wikipedia:
        strings = create_strings_from_wikipedia(
            args.length, args.count, args.language)
    elif args.input_file != '':
        strings = create_strings_from_file(
            args.input_file, args.count, args.length)
    elif args.random_sequences:
        strings = create_strings_randomly(args.length, args.random, args.count,
                                          args.include_letters, args.include_numbers, args.include_symbols, args.language)
        # Set a name format compatible with special characters automatically if they are used
        if args.include_symbols or True not in (args.include_letters, args.include_numbers, args.include_symbols):
            args.name_format = 2
    elif args.random_sequences_from_font:
        strings = create_strings_from_fonts(fonts_arr)
    elif args.random_sequences_sjnk:
        strings = random_sequences_sjnk(font_charsets)
    elif args.random_latin_sjnk:
        strings = random_latin(font_charsets)
    elif args.random_space:
        strings = random_space(font_charsets)
    elif args.random_latin_space:
        strings = random_latin_space(font_charsets)
    elif args.check_font:
        strings = gen_check_font(font_charsets)
    elif args.random_one_character:
        fonts_arr, strings = gen_one_character(fonts_dict)
    else:
        # Creating word list
        lang_dict = load_dict(args.language)

        strings = create_string_from_dict_with_random_chars(
            args.length, args.random, args.count, lang_dict)

    strings = [''.join([c for c in text if c in charset])
               for text, charset in zip(strings, font_charsets)]
    # strings = [s.strip() for s in strings if len(s.strip()) > 1]

    if os.path.isdir('./logs/') is False:
        os.system("mkdir logs")

    string_count = len(strings)
    print("Generated", string_count, "string.")
    if string_count < 10:
        [print('\t' + each_string) for each_string in strings]
    else:
        [print('\t' + each_string) for each_string in strings[:5]]
        print('\t...')

    # print(strings)

    print_text("./logs/src-train.txt", ['{}_{}.{}\n'.format(
        args.prefix, str(index), args.extension) for index in range(string_count)])
    print_text("./logs/tgt-train.txt", ['{}\n'.format(x) for x in strings])
    print_text("./logs/tgt-fonts.txt", ['{}\n'.format(x) for x in fonts_arr])

    # exit()
    print('Generating images from the string ...')
    p = multiprocessing.Pool(args.thread_count)
    for _ in tqdm(
            p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    fonts_arr,
                    [args.output_dir] * string_count,
                    [args.format] * string_count,
                    # [random.randint(args.format, args.format + 40) for x in range(string_count)],
                    [args.extension] * string_count,
                    [args.skew_angle] * string_count,
                    [args.random_skew] * string_count,
                    [args.blur] * string_count,
                    [args.background] * string_count,
                    [args.distorsion] * string_count,
                    [args.distorsion_orientation] * string_count,
                    [args.handwritten] * string_count,
                    [args.name_format] * string_count,
                    [-1] * string_count,
                    [args.prefix] * string_count,
                    [args.random_crop] * string_count,
                    [args.debug] * string_count
                )
            ), total=args.count):
        pass
    p.terminate()
    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                f.write("{} {}\n".format(file_name, strings[i]))
    print('Generated', args.count, 'images to', args.output_dir)

if __name__ == '__main__':
    main()
