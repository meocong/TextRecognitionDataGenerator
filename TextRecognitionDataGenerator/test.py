import pickle
fonts_dict = pickle.load(open("font_dict.pkl", "rb"))
# font_charsets = [fonts_dict[font] for font in fonts_arr]
print(fonts_dict)