from .model import Model # was from models.model import Model
from tensorflow.keras import Sequential, layers
# was from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model

        # Initialize the model as a Sequential model
        self.model = Sequential([
            # Rescale layer to normalize pixel values
            Rescaling(1./255, input_shape=input_shape),

            # Convolutional layers with ReLU activation
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # Flatten the 3D output to 1D
            Flatten(),

            # Fully connected layers
            Dense(128, activation='relu'),
            Dropout(0.5),

            Dense(64, activation='relu'),

            # Output layer with softmax activation for multi-class classification
            Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        
        # Compile the model with an optimizer, loss function, and evaluation metric
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # Using Adam optimizer
            loss='categorical_crossentropy',     # Appropriate for multi-class classification
            metrics=['accuracy']                 # Tracking accuracy during training
        )

# if __name__ == "__main__":
#     input_shape = (150, 150, 3)  # Example input shape
#     categories_count = 3  # Number of categories (e.g., 3 facial expressions)

#     model = BasicModel(input_shape, categories_count)
#     model.print_summary()  # This will print out the model's architecture