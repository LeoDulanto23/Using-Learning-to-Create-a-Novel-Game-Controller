from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
from config import train_directory, test_directory, image_size, batch_size, validation_split

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )

    # Adding the Rescaling layer to the dataset pipeline
    train_dataset = train_dataset.map(lambda x, y: (Rescaling(1./255)(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (Rescaling(1./255)(x), y))


    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )


    # Rescaling test data as well
    test_dataset = test_dataset.map(lambda x, y: (Rescaling(1./255)(x), y))

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # Your code replaces this by loading the dataset
    # you can use image_dataset_from_directory, similar to how the _split_data function is using it
    train_dataset, validation_dataset, test_dataset = None, None, None
    # Example:
    # train_dataset = image_dataset_from_directory(transfer_train_directory, ...)
    # Add rescaling if needed using map with Rescaling(1./255)
    return train_dataset, validation_dataset, test_dataset