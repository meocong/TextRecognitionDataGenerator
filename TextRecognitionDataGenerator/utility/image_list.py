import glob
import cv2


def print_text(file, list_):
    f = open(file, 'w')
    f.writelines(list_)


images = [x.split("/")[-1]+"\n" for x in glob.glob("../images/*.jpg")]
print_text("image_list.txt", images)

for index, image in enumerate(images):
    img = cv2.imread(image, 0)
    cv2.imshow(str(index), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
