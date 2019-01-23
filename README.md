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
Edit [config.yaml](./TextRecognitionDataGenerator/config.yaml) to config the script. Then run `python3 run.py` to start generating.

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
