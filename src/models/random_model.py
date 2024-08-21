from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        # very similar to transfered_model.py, the only difference is that you should randomize the weights
        # load your basic model with keras's load_model function
        # freeze the weights of the loaded model to make sure the training doesn't affect them
        # (check the number of total params, trainable params and non-trainable params in your summary generated by train_transfer.py)
        # randomize the weights of the loaded model, possibly by using _randomize_layers
        # use this model by removing the last layer, adding dense layers and an output layer
        # Load the basic model
        base_model = load_model(basic_model.keras)  # Replace with the correct path
        for layer in base_model.layers:
            layer.trainable = False
        # Randomize the weights of the layers in the base model
        self._randomize_layers(base_model)
        # Create a new Sequential model
        self.model = Sequential()
        # Add the base model
        self.model.add(base_model)
        # Optionally remove the last layer if needed
        self.model.layers.pop()
        # Add new layers on top of the base model
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(categories_count, activation='softmax'))
        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        # Compile the model with an optimizer, loss function, and metrics
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        pass

    @staticmethod
    def _randomize_layers(model):
        # Your code goes here
        # you can write a function here to set the weights to a random value
        # use this function in _define_model to randomize the weights of your loaded model
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                # Randomize the weights (kernels) and biases
                new_kernel = layer.kernel_initializer(shape=layer.kernel.shape)
                new_bias = layer.bias_initializer(shape=layer.bias.shape)
                layer.set_weights([new_kernel, new_bias])
        pass

