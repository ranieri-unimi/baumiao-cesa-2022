#check for corrupted files (method 1)
from pathlib import Path
import imghdr
import os
from PIL import Image
import tensorflow as tf

classes = ['Cat','Dog']
image_extensions = [".png", ".jpg"] # add there all your images file extensions

for item in classes:
    data_dir = f'kagglecatsanddogs_5340/PetImages/{item}'
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                # remove image
                os.remove(filepath)
                print(f"successfully removed {filepath}")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                try:
                    img = Image.open(filepath) # open the image file
                    img.verify() # verify that it is, in fact an image
                    # print('valid image')
                except Exception:
                    print('Bad file:', filepath)


# method 2
import glob
img_paths = glob.glob(os.path.join('kagglecatsanddogs_5340/PetImages/Dog','*/*.*')) # assuming you point to the directory containing the label folders.
bad_paths = []
for image_path in img_paths:
    try:
        img_bytes = tf.io.read_file(image_path)
        decoded_img = tf.decode_image(img_bytes)
    except Exception:
        print(f"Found bad path {image_path}")
        bad_paths.append(image_path)

    # print(f"{image_path}: OK")

print("BAD PATHS:")
for bad_path in bad_paths:
    print(f"{bad_path}")


# method 3
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("kagglecatsanddogs_5340/PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
            print("Deleted %d images" % num_skipped)

def normalize(x,y):
    x = tf.cast(x,tf.float32) / 255.0
    return x, y

def convert_to_categorical(input):
    if input == 1:
        return "Dog"
    else:
        return "Cat"

def to_list(ds):
    ds_list = []
    for sample in ds:
        image, label = sample
        ds_list.append((image, label))
    return ds_list

# load dataset
directory = 'archive/PetImages'
ds_train = tf.keras.utils.image_dataset_from_directory(
directory,
labels='inferred',
label_mode='binary',
color_mode='rgb',
batch_size=1,
shuffle=False,
validation_split=0.3,
subset='training',
image_size=(180,180)
)
ds_test = tf.keras.utils.image_dataset_from_directory(
directory,
labels='inferred',
label_mode='binary',
color_mode='rgb',
batch_size=1,
shuffle=False,
validation_split=0.3,
subset='validation',
image_size=(180,180)
)

# normalize data
ds_train.map(normalize)
ds_test.map(normalize)

# plot 10 random images from training set
num = len(ds_train)
ds_train_list = to_list(ds_train)
for i in range(1,11):
random_index = np.random.randint(num)
img, label = ds_train_list[random_index]
label = convert_to_categorical(np.array(label))
img = np.reshape(img,(300,300,3))
plt.subplot(2,5,i)
plt.imshow(img)
plt.title(label)
plt.savefig('figures/example_images.png')