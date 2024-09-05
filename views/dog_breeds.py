import streamlit as st

st.title('🐶 End-End dog breed identification Model using Tensorflow')

st.markdown("""
            ## Problem

Identifying the breed of a dog given an image of a dog.

## Data

The data I am using is from Kaggle's dog breed identification competition
https://www.kaggle.com/c/dog-breed-identification/data

## Evaluation

Evaluation is a file with prediction probabilities for each dog breed of each test image
https://www.kaggle.com/competitions/dog-breed-identification/overview

## Features

Some information of data:

* We're dealing with images (unstructured data) so it's probably best we use deep learning or transfer learning
* There are 120 breeds of dogs i.e 120 different classes.
* There are 10,000+ images in training set
* There are 10,100+ images in test set
""")
st.write('\n')
st.code(
    """
    import tensorflow as tf
    import tensorflow_hub as hub
    print("Tf hub version_", hub.__version__)
    print("Tf version_", tf.__version__)
    # check for GPU
    print("GPU", "available (YESS)" if tf.config.list_physical_devices("GPU") else "not available")
    """, language="python"
)
st.write('\n')
st.markdown("### loading data and turning them into tensors")
st.write('\n')
st.code(
    """
    import pandas as pd
    labels_csv = pd.read_csv("/content/drive/MyDrive/Dog_breed_finder/labels.csv")
    print(labels_csv.describe())
    """, language="python"
)
st.image('./assets/desc.jpg')
st.code(
    '''
    # view a image
    from IPython.display import Image
    Image("/content/drive/MyDrive/Dog_breed_finder/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg")    
''', language="python"
)
st.image('./assets/im1.jpeg')
st.code(
    '''
    # Turn every label into a boolean array
    boolean_labels = [label == unique_breeds for label in labels]
    boolean_labels[:2]    
''', language="python"
)
st.markdown(
    '''
    [array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False]),
 array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False])] 
'''
)
st.code('len(boolean_labels)', language="python")
st.markdown('10222')
st.code(
    '''
    # Example: Turning boolean array into integers
    print(labels[0]) # original label
    print(np.where(unique_breeds == labels[0])) # index where label occurs
    print(boolean_labels[0].argmax()) # index where label occurs in boolean array
    print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs    
''', language='python'
)
st.markdown(
    """
    boston_bull
(array([19]),)
19
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0]    
"""
)
st.write('\n')
st.markdown(
    '''## Creating our own validation set'''
)
st.code(
    '''
    # Setup X & y variables
    X = filenames
    y = boolean_labels

    # Set number of images to use for experimenting
    NUM_IMAGES = 1000 #@param {type:"slider", min:1000, max:10000, step:1000}
''', language="python"
)
st.code(
    '''
    # Let's split our data into train and validation sets
    from sklearn.model_selection import train_test_split

    # Split them into training and validation of total size NUM_IMAGES
    X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

    len(X_train), len(y_train), len(X_val), len(y_val)    
'''
)
st.code(
    '''
    # Let's have a look at the training data
    X_train[:5], y_train[:2]    
''', language="python"
)
st.markdown(
    """
    (['/content/drive/MyDrive/Dog_breed_finder/train/00bee065dcec471f26394855c5c2f3de.jpg',
  '/content/drive/MyDrive/Dog_breed_finder/train/0d2f9e12a2611d911d91a339074c8154.jpg',
  '/content/drive/MyDrive/Dog_breed_finder/train/1108e48ce3e2d7d7fb527ae6e40ab486.jpg',
  '/content/drive/MyDrive/Dog_breed_finder/train/0dc3196b4213a2733d7f4bdcd41699d3.jpg',
  '/content/drive/MyDrive/Dog_breed_finder/train/146fbfac6b5b1f0de83a5d0c1b473377.jpg'],
 [array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False,  True,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False]),
  array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False,  True, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False])])    
"""
)
st.write('\n')
st.markdown(
    """
    ## Preprocess images into tensors
    * Take an image filepath as input
    * Use TensorFlow to read the file and save it to a variable, image
    * Turn our image (a jpg) into Tensors
    * Normalize our image (convert color channel values from from 0-255 to 0-1).
    * Resize the image to be a shape of (224, 224)
    * Return the modified image    
"""
)
st.code(
    '''
    # Convert image to NumPy array
    from matplotlib.pyplot import imread
    image = imread(filenames[42])
    image.shape        
''', language='python'
)
st.code(
    '''
    # turn image into a tensor
    tf.constant(image)[:2]        
''', language='python'
)
st.write('\n')
st.markdown(
    '''
    ## Now we've seen what an image looks like as a Tensor, let's make a function to preprocess them.

    We'll create a function to:

    * Take an image filepath as input
    * Use TensorFlow to read the file and save it to a variable, `image`
    * Turn our `image` (a jpg) into Tensors
    * Normalize our `image` (convert color channel values from from 0-255 to 0-1).
    * Resize the `image` to be a shape of (224, 224)
    * Return the modified `image`

    More information on loading images in TensorFlow can be seen here: https://www.tensorflow.org/tutorials/load_data/images    
'''
)
st.code(
    '''
    # Define image size
    IMG_SIZE = 224

    # Create a function for preprocessing images
    def process_image(image_path, img_size=IMG_SIZE):
        """
        Takes an image file path and turns the image into a Tensor.
        """
        # Read in an image file
        image = tf.io.read_file(image_path)
        # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        image = tf.image.decode_jpeg(image, channels=3)
        # Convert the colour channel values from 0-255 to 0-1 values
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image to our desired value (224, 224)
        image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

        return image    
'''
)
st.markdown(
    '''
    ## Turning our data into batches
    Why turn our data into batches?

    Let's say you're trying to process 10,000+ images in one go... they all might not fit into memory.

    So that's why we do about 32 (this is the batch size) images at a time (you can manually adjust the batch size if need be).

    In order to use TensorFlow effectively, we need our data in the form of Tensor tuples which look like this: `(image, label)`.
'''
)
st.code(
    '''
    # Create a simple function to return a tuple (image, label)
    def get_image_label(image_path, label):
        """
        Takes an image file path name and the assosciated label,
        processes the image and reutrns a typle of (image, label).
        """
        image = process_image(image_path)
        return image, label
    
    # Demo of the above
    (process_image(X[42]), tf.constant(y[42]))
''', language='python'
)
st.markdown(
    '''
    Now we've got a way to turn our data into tuples of Tensors in the form: `(image, label)`, let's make a function to turn all of our data (`X` & `y`) into batches!
'''
)
st.code(
    '''
    # Define the batch size, 32 is a good start
    BATCH_SIZE = 32

    # Create a function to turn data into batches
    def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
        """
        Creates batches of data out of image (X) and label (y) pairs.
        Shuffles the data if it's training data but doesn't shuffle if it's validation data.
        Also accepts test data as input (no labels).
        """
        # If the data is a test dataset, we probably don't have have labels
        if test_data:
            print("Creating test data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
            data_batch = data.map(process_image).batch(BATCH_SIZE)
            return data_batch

        # If the data is a valid dataset, we don't need to shuffle it
        elif valid_data:
            print("Creating validation data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths
                                                    tf.constant(y))) # labels
            data_batch = data.map(get_image_label).batch(BATCH_SIZE)
            return data_batch

        else:
            print("Creating training data batches...")
            # Turn filepaths and labels into Tensors
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                    tf.constant(y)))
            # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
            data = data.shuffle(buffer_size=len(X))

            # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)
            data = data.map(get_image_label)

            # Turn the training data into batches
            data_batch = data.batch(BATCH_SIZE)
        return data_batch
    
    # Create training and validation data batches
    train_data = create_data_batches(X_train, y_train)
    val_data = create_data_batches(X_val, y_val, valid_data=True)
''', language='python'
)
st.markdown(
    '''
    ## Visualizing Data Batches
    Our data is now in batches, however, these can be a little hard to understand/comprehend, let's visualize them!    
'''
)
st.code(
    '''
    import matplotlib.pyplot as plt

    # Create a function for viewing images in a data batch
    def show_25_images(images, labels):
        """
        Displays a plot of 25 images and their labels from a data batch.
        """
        # Setup the figure
        plt.figure(figsize=(10, 10))
        # Loop through 25 (for displaying 25 images)
        for i in range(25):
            # Create subplots (5 rows, 5 columns)
            ax = plt.subplot(5, 5, i+1)
            # Display an image
            plt.imshow(images[i])
            # Add the image label as the title
            plt.title(unique_breeds[labels[i].argmax()])
            # Turn the grid lines off
            plt.axis("off")
''', language='python'
)
st.markdown(
    '''
    ## Building a model
    Before we build a model, there are a few things we need to define:

    * The input shape (our images shape, in the form of Tensors) to our model.
    * The output shape (image labels, in the form of Tensors) of our model.
    * The URL of the model we want to use from TensorFlow Hub - https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
'''
)
st.code(
    '''
    # Setup input shape to the model
    INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

    # Setup output shape of our model
    OUTPUT_SHAPE = len(unique_breeds)

    # Setup model URL from TensorFlow Hub
    MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
'''
)
st.markdown(
    '''
    Now we've got our inputs, outputs and model ready to go. Let's put them together into a Keras deep learning model!

    Knowing this, let's create a function which:

    * Takes the input shape, output shape and the model we've chosen as parameters.
    * Defines the layers in a Keras model in sequential fashion (do this first, then this, then that).
    * Compiles the model (says it should be evaluated and improved).
    * Builds the model (tells the model the input shape it'll be getting).
    * Returns the model.

    All of these steps can be found here: https://www.tensorflow.org/guide/keras/overview
'''
)
st.code(
    '''
    # Create a function which builds a Keras model
    def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
        print("Building model with:", MODEL_URL)

        # Setup the model layers
        model = tf.keras.Sequential([
            hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
            tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                                activation="softmax") # Layer 2 (output layer)
        ])

        # Compile the model
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

        # Build the model
        model.build(INPUT_SHAPE)

        return model
    
    model = create_model()
    model.summary()
''', language='python'
)
st.markdown(
    '''
    ### Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
    - Model: "sequential"
    - Layer (type)                Output Shape              Param   
    - keras_layer (KerasLayer)    (None, 1001)              5432713     
    - dense (Dense)               (None, 120)               120240    
    ---
    - Total params: 5552953 (21.18 MB)
    - Trainable params: 120240 (469.69 KB)
    - Non-trainable params: 5432713 (20.72 MB)
    ---
'''
)
st.markdown(
    '''
    ## Creating callbacks
    Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.

    We'll create two callbacks, one for TensorBoard which helps track our models progress and another for early stopping which prevents our model from training for too long.
'''
)
st.markdown(
    '''
    ## TensorBoard Callback
    To setup a TensorBoard callback, we need to do 3 things:

    * Load the TensorBoard notebook extension ✅
    * Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's fit() function. ✅
    * Visualize our models training logs with the %tensorboard magic function (we'll do this after model training).

    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
'''
)
st.code(
    '''
    import datetime

    # Create a function to build a TensorBoard callback
    def create_tensorboard_callback():
        # Create a log directory for storing TensorBoard logs
        logdir = os.path.join("drive/My Drive/Dog Vision/logs",
                                # Make it so the logs get tracked whenever we run an experiment
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return tf.keras.callbacks.TensorBoard(logdir)
    ''', language='python'
)
st.markdown(
    '''
    ## Early Stopping Callback
    Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving.

    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
'''
)
st.code(
    '''
    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                    patience=3)    
''', language='python'
)
st.markdown(
    '''
    Let's create a function which trains a model.

    * Create a model using `create_model()`
    * Setup a TensorBoard callback using `create_tensorboard_callback()`
    * Call the `fit()` function on our model passing it the training data, validation data, number of epochs to train for `(NUM_EPOCHS)` and the callbacks we'd like to use
    * Return the model    
'''
)
st.code(
    '''
    # Build a function to train and return a trained model
    def train_model():
        """
        Trains a given model and returns the trained version.
        """
        # Create a model
        model = create_model()

        # Create new TensorBoard session everytime we train a model
        tensorboard = create_tensorboard_callback()

        # Fit the model to the data passing it the callbacks we created
        model.fit(x=train_data,
                    epochs=NUM_EPOCHS,
                    validation_data=val_data,
                    validation_freq=1,
                    callbacks=[tensorboard, early_stopping])
        # Return the fitted model
        return model
''', language='python'
)
st.code(
    '''
    # Make predictions on the validation data (not used to train on)
    predictions = model.predict(val_data, verbose=1)
    predictions    
''', language='python'
)
st.markdown(
    '''
    7/7 [==============================] - 1s 82ms/step
    array([[3.3672882e-04, 2.6400160e-04, 3.3767702e-04, ..., 4.1428625e-04,
            3.2165994e-05, 3.1243530e-03],
        [5.1022191e-03, 1.0333934e-03, 6.4172111e-02, ..., 2.0356418e-03,
            2.6253765e-03, 4.5186235e-04],
        [1.7507989e-05, 2.8676443e-05, 3.8780786e-06, ..., 3.7129932e-05,
            2.1261997e-04, 1.6737473e-04],
        ...,
        [1.7152402e-06, 3.7766280e-05, 1.3605044e-05, ..., 2.5114594e-05,
            6.2662330e-05, 8.4599676e-05],
        [5.2315900e-03, 4.4860557e-04, 1.6435220e-04, ..., 5.3224253e-04,
            6.0256873e-05, 1.5583080e-02],
        [1.9731569e-04, 6.1917897e-05, 1.9549704e-03, ..., 2.9542071e-03,
            1.2823994e-03, 1.3980542e-05]], dtype=float32)    
'''
)
st.code(
    '''
    # First prediction
    index = 42
    print(predictions[index])
    print(f"Max value (probability of prediction): {np.max(predictions[index])}")
    print(f"Sum: {np.sum(predictions[index])}")
    print(f"Max index: {np.argmax(predictions[index])}")
    print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")
''', language='python'
)
st.markdown(
    '''    
    [6.99749071e-05 3.75268392e-05 2.53315593e-05 2.32899692e-05
    4.46435151e-04 1.43183515e-05 1.32108587e-04 4.09331959e-04
    6.69416506e-03 3.61800678e-02 2.18954119e-05 3.61140474e-06
    1.82684205e-04 2.19344744e-03 7.62850977e-04 7.05648970e-04
    9.70163273e-06 1.37182287e-04 2.10635102e-04 1.29106033e-04
    3.14165991e-05 3.09754745e-04 1.17811232e-05 1.34849615e-05
    4.20762366e-03 2.48350880e-05 3.18074563e-05 4.74966728e-05
    1.46850536e-04 3.42293060e-05 6.40612870e-06 6.32427473e-05
    3.45292683e-05 3.79305056e-05 1.67694416e-05 2.77101408e-05
    9.02506945e-05 6.21083091e-05 1.30927519e-05 2.07220793e-01
    4.69435472e-05 1.92239677e-05 5.58238244e-03 2.52855898e-06
    1.51557018e-04 3.44698783e-05 9.52121554e-05 3.73387971e-04
    1.47167593e-05 1.97018890e-04 8.34637831e-05 4.10017419e-05
    7.59320537e-05 4.45041049e-04 1.02949225e-05 3.50873597e-04
    1.34196933e-04 1.82165939e-04 2.25779550e-05 1.50696302e-04
    4.11289839e-05 2.32945808e-04 5.34803576e-06 7.13686459e-05
    1.35832423e-04 1.89598722e-05 2.87908315e-05 2.21572700e-05
    1.44913531e-04 3.29967661e-05 1.02506659e-04 1.32736550e-05
    5.23002309e-05 1.73811015e-04 4.04230850e-05 1.00554294e-04
    1.40265605e-04 2.19815902e-05 2.66626394e-05 3.39290389e-04
    7.25332347e-06 3.30831645e-05 7.39082243e-05 2.82751804e-04
    6.24206674e-04 6.34275129e-05 2.12015235e-04 1.21090875e-06
    2.12875784e-05 6.58257748e-04 1.14519891e-04 6.09805102e-06
    7.32974790e-04 4.36415503e-05 5.47778518e-06 5.79168664e-05
    3.16420519e-05 1.58218936e-05 4.65946214e-05 7.43273122e-05
    6.92084213e-05 1.36251867e-04 6.38622150e-05 1.29082564e-05
    7.88068355e-05 3.60209197e-05 1.29559194e-04 4.46106533e-05
    3.69037007e-05 1.19501063e-04 1.32281930e-04 1.43722142e-03
    7.18982847e-05 7.22467780e-01 5.18489571e-04 1.71695327e-04
    3.72093855e-05 3.87679138e-05 6.77510863e-04 3.24033281e-05]
    - Max value (probability of prediction): 0.7224677801132202
    - Sum: 0.9999998807907104
    - Max index: 113
    - Predicted label: walker_hound
'''
)
st.code(
    '''
    # Turn prediction probabilities into their respective label (easier to understand)
    def get_pred_label(prediction_probabilities):
        """
        Turns an array of prediction probabilities into a label.
        """
        return unique_breeds[np.argmax(prediction_probabilities)]

        # Get a predicted label based on an array of prediction probabilities
        pred_label = get_pred_label(predictions[81])
        pred_label
''', language='python'
)
st.code(
    '''
    # Create a function to unbatch a batch dataset
    def unbatchify(data):
        """
        Takes a batched dataset of (image, label) Tensors and reutrns separate arrays
        of images and labels.
        """
        images = []
        labels = []
        # Loop through unbatched data
        for image, label in data.unbatch().as_numpy_iterator():
            images.append(image)
            labels.append(unique_breeds[np.argmax(label)])
        return images, labels

        # Unbatchify the validation data
        val_images, val_labels = unbatchify(val_data)
        val_images[0], val_labels[0]    
''', language='python'
)
st.markdown(
    '''
    Now we've got ways to get get:

    * Prediction labels
    * Validation labels (truth labels)
    * Validation images
    * Let's make some function to make these all a bit more visaulize.

    We'll create a function which:

    * Takes an array of prediction probabilities, an array of truth labels and an array of images and an integer. ✅
    * Convert the prediction probabilities to a predicted label. ✅
    * Plot the predicted label, its predicted probability, the truth label and the target image on a single plot. ✅
'''
)
st.code(
    '''
    def plot_pred(prediction_probabilities, labels, images, n=1):
        """
        View the prediction, ground truth and image for sample n
        """
        pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

        # Get the pred label
        pred_label = get_pred_label(pred_prob)

        # Plot image & remove ticks
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        # Change the colour of the title depending on if the prediction is right or wrong
        if pred_label == true_label:
            color = "green"
        else:
            color = "red"

        # Change plot title to be predicted, probability of prediction and truth label
        plt.title("{} {:2.0f}% {}".format(pred_label,
                                            np.max(pred_prob)*100,
                                            true_label),
                                            color=color)
    
    plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,
          n=77)
''', language='python'
)
st.image('./assets/im2.png')
st.markdown(
    '''
    Now we've got one function to visualize our models top prediction, let's make another to view our models top 10 predictions.

    This function will:

    * Take an input of prediction probabilities array and a ground truth array and an integer ✅
    * Find the prediction using get_pred_label() ✅
    * Find the top 10:
    * Prediction probabilities indexes ✅
    * Prediction probabilities values ✅
    * Prediction labels ✅
    * Plot the top 10 prediction probability values and labels, coloring the true label green ✅
'''
)
st.code(
    '''
    def plot_pred_conf(prediction_probabilities, labels, n=1):
        """
        Plus the top 10 highest prediction confidences along with the truth label for sample n.
        """
        pred_prob, true_label = prediction_probabilities[n], labels[n]

        # Get the predicted label
        pred_label = get_pred_label(pred_prob)

        # Find the top 10 prediction confidence indexes
        top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
        # Find the top 10 prediction confidence values
        top_10_pred_values = pred_prob[top_10_pred_indexes]
        # Find the top 10 prediction labels
        top_10_pred_labels = unique_breeds[top_10_pred_indexes]

        # Setup plot
        top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                            top_10_pred_values,
                            color="grey")
        plt.xticks(np.arange(len(top_10_pred_labels)),
                    labels=top_10_pred_labels,
                    rotation="vertical")

        # Change color of true label
        if np.isin(true_label, top_10_pred_labels):
            top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
        else:
            pass
    
    plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)
''', language='python'
)
st.image('./assets/im3.png')
st.markdown(
    '''
    Now we've got some function to help us visualize our predictions and evaluate our modle, let's check out a few.    
'''
)
st.code(
    '''
    # Let's check out a few predictions and their different values
    i_multiplier = 20
    num_rows = 3
    num_cols = 2
    num_images = num_rows*num_cols
    plt.figure(figsize=(10*num_cols, 5*num_rows))
    for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_pred(prediction_probabilities=predictions,
                labels=val_labels,
                images=val_images,
                n=i+i_multiplier)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_pred_conf(prediction_probabilities=predictions,
                    labels=val_labels,
                    n=i+i_multiplier)
    plt.tight_layout(h_pad=1.0)
    plt.show()
''',language='python'
)
st.image('./assets/im4.png')
st.write('\n')
st.markdown(
    "### That's about it for this project for further info please visit: [link](https://github.com/Arraj2611/ml_projects/blob/main/dog_breed_identification.ipynb)"
)