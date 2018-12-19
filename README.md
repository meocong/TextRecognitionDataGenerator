# TextRecognitionDataGenerator

A synthetic data generator for text recognition

## Dependencies

```
Python 3.X
OpenCV 3.2 (It probably works with 2.4)
Pillow
Numpy
Requests
BeautifulSoup
tqdm
pyblur (don't use version on pypi)
```

 You can simply run `pip3 install -r requirements.txt`.
 Then run `./install_pyblur.sh` to install our custom version or `pyblur`

Then, download these additional files: 
- Background pictures: https://drive.google.com/file/d/1Ck62gjFVuQtidoVd9I8vbQzkI0N4WIdS/view?usp=sharing
- Fonts: https://drive.google.com/file/d/1oYlY_DLk4W4PZA5CLxRlNczBpQMTa6bu/view?usp=sharing

Extract both of it to `TextRecognitionDataGenerator` folder.

## How to run
Go to `./TextRecognitionDataGenerator`, then run `python3 run.py` with these following arguments

### Command line arguments

| Long Arg                     | Short Arg   | Description                                                                                                                                                 | Default |
|------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| --output_dir                 |             | The output directory                                                                                                                                        | images/ |
| --input_file                 | -i          | When set, this argument uses a specified text file as source for the text                                                                                   |         |
| --language                   | -l          | The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).                                                       | en      |
| --count                      | -c          | The number of images to be created.                                                                                                                         | 1000    |
| --random_sequences           | -rs         | Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.    | False   |
| --random_sequences_from_font | -rsff       | Use random sequences by characters in random fonts as the source text for the generation.                                                                   | False   |
| --random_one_character       | -rdo        | Use random sequences by characters in random fonts as the source text for the generation.                                                                   | False   |
| --random_sequences_sjnk      | -sjnk       | Generate data for sjnk project                                                                                                                              | False   |
| --random_latin_sjnk          | -sjnk_latin | Generate data for sjnk project                                                                                                                              | False   |
| --random_latin_space         | -rdlt       | Generate data for sjnk project                                                                                                                              | False   |
| --include_letters            | -let        | Define if random sequences should contain letters. Only works with -rs                                                                                      | False   |
| --include_symbols            | -sym        | Define if random sequences should contain symbols. Only works with -rs                                                                                      | False   |
| --length                     | -w          | Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length                              | 1       |
| --random                     | -r          | Define if the produced string will have variable word count (with --length being the maximum)                                                               | False   |
| --random_space               | -rp         | Random Space                                                                                                                                                | False   |
| --format                     | -f          | Define the height of the produced images                                                                                                                    | 32      |
| --thread_count               | -t          | Define the number of thread to use for image generation                                                                                                     | 1       |
| --extension                  | -e          | Define the extension to save the image with                                                                                                                 | jpg     |
| --prefix                     | -pre        | Define the extension to save the image with                                                                                                                 |         |
| --skew_angle                 | -k          | Define skewing angle of the generated text. In positive degrees                                                                                             | 0       |
| --random_skew                | -rk         | When set, the skew angle will be randomized between the value set with -k and it's opposite                                                                 | False   |
| --use_wikipedia              | -wk         | Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s                                                                | False   |
| --blur                       | -bl         | Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius                                                                  | 0       |
| --random_blur                | -rbl        | When set, the blur radius will be randomized between 0 and -bl.                                                                                             | False   |
| --background                 | -b          | Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures                                                      | 0       |
| --handwritten                | -hw         | Define if the data will be \"handwritten\" by an RNN                                                                                                        |         |
| --name_format                | -na         | Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings | 0       |
| --distorsion                 | -d          | Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random                                              | 0       |
| --distorsion_orientation     | -do         | Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both                       | 0       |

### Example

Run `python run.py -w 5 -f 64`. You get 1000 randomly generated images with random text on them like:

![1](samples/1.jpg "1")
![2](samples/2.jpg "2")
![3](samples/3.jpg "3")
![4](samples/4.jpg "4")
![5](samples/5.jpg "5")

Random skewing: Add `-k` and `-rk` (`python run.py -w 5 -f 64 -k 5 -rk`)

![6](samples/6.jpg "6")
![7](samples/7.jpg "7")
![8](samples/8.jpg "8")
![9](samples/9.jpg "9")
![10](samples/10.jpg "10")

Add `-bl` and `-rbl` to get gaussian blur on the generated image with user-defined radius (here 0, 1, 2, 4):

![11](samples/11.jpg "0")
![12](samples/12.jpg "1")
![13](samples/13.jpg "2")
![14](samples/14.jpg "4")

Add `-b` to define one of the three available backgrounds: gaussian noise (0), plain white (1), quasicrystal (2) or picture (3).

![15](samples/15.jpg "0")
![16](samples/16.jpg "1")
![17](samples/17.jpg "2")
![23](samples/23.jpg "3")

When using picture background (3). A picture from the pictures/ folder will be randomly selected and the text will be written on it.

Handwritten text: Add `-hw`! (Experimental)

![18](samples/18.jpg "0")
![19](samples/19.jpg "1")
![20](samples/20.jpg "2")
![21](samples/21.jpg "3")
![22](samples/22.jpg "4")

It uses a Tensorflow model trained using [this excellent project](https://github.com/Grzego/handwriting-generation) by Grzego.

**The project does not require TensorFlow to run if you aren't using this feature**

Add distorsion to the generated text: `-d` and `-do`

![23](samples/24.jpg "0")
![24](samples/25.jpg "1")
![25](samples/26.jpg "2")

The text is chosen at random in a dictionary file (that can be found in the *dicts* folder) and drawn on a white background made with Gaussian noise. The resulting image is saved as [text]\_[index].jpg

### Create images with Chinese (both simplified and traditional) text

`python run.py -l cn -c 1000 -w 5`!

Unfortunately I do not speak Chinese so you may have to edit `texts/cn.txt` to include some meaningful words instead of random glyphs.

Here are examples of what I could make with it:

Traditional:

![27](samples/27.jpg "0")

Simplified:

![28](samples/28.jpg "1")

## Can I add my own font?

Yes, the script picks a font at random from the *fonts* directory. 

|||
|----:|:-----|
| fonts/latin | English, French, Spanish, German |
| fonts/cn | Chinese |
|||

Simply add / remove fonts until you get the desired output.

If you want to add a new non-latin language, the amount of work is minimal.

1. Create a new folder with your language two-letters code
2. Add a .ttf font in it
3. Edit `run.py` to add an if statement in `load_fonts()`
4. Add a text file in `dicts` with the same two-letters code
5. Run the tool as you normally would but add `-l` with your two-letters code

It only supports .ttf for now.

## Contributing

1. Create an issue describing the feature you'll be working on
2. Code said feature
3. Create a pull request

## Feature request & issues

If anything is missing, unclear, or simply not working, open an issue on the repository.

## Future improvement
- Better background generation
- Better handwritten text generation
- More customization parameters (mostly regarding background)
