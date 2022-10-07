from PIL import UnidentifiedImageError
import colorsys

img = Image.open(uri.format(CAT, 35))
lx, ly = img.size
img

pxl = img.load()
a = [[((np.array(pxl[x, y]) ** 2) / (255**2)) for x in range(lx)] for y in range(ly)]
plt.imshow(a, interpolation="none")
plt.show()

# img_paths = glob.glob(os.path.join(<path_to_dataset>,'*/*.*'))
# assuming you point to the directory containing the label folders.

bad_paths = []

for image_path in img_paths:
    image_path = uri.format(CAT, 35)
    try:
        img_bytes = tf.io.read_file(path)
        decoded_img = tf.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError as e:
        print(f"Found bad path {image_path}...{e}")
        bad_paths.append(image_path)

    print(f"{image_path}: OK")

print("BAD PATHS:")
for bad_path in bad_paths:
    print(f"{bad_path}")

    model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPool2D(2, 2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function
model.add(keras.layers.Dense(512, activation="relu"))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog
# Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# steps_per_epoch = train_imagesize/batch_size

model.fit_generator(
    train_dataset, steps_per_epoch=250, epochs=10, validation_data=test_dataset
)
