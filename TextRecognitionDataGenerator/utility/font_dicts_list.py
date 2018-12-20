import pickle


# print("Font charsets: ")
# font_charsets = [fonts_dict[font] for font in fonts_arr]
# print(font_charsets)

print("Font dicts: ")
fonts_dict = pickle.load(open("../fonts/jp/font_dict.pkl", "rb"))
print(fonts_dict)
