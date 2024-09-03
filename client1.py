import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import flwr as fl
from model_dataDiv import get_data
from tensorflow.keras.callbacks import EarlyStopping


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Compile model with Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Monitor validation loss, stop after 3 epochs of no improvement



def my_model():
    
    model = tf.keras.models.Sequential([
        # Dense layers for classification
        tf.keras.layers.Dense(units=128, activation='relu', input_dim=16),
        tf.keras.layers.Dropout(rate=0.2),  # Dropout for regularization
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),  # Dropout for regularization
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),  # Dropout for regularization
        # Final output layer with softmax activation for 12 classes
        tf.keras.layers.Dense(units=12, activation='softmax')
        ])
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    return model


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = my_model()
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
            callbacks=[early_stopping],
        )
       
    


        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]
        

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:    

    # Load and compile Keras model
    model = my_model()
    client_id=1
    # Load a subset of data to simulate the local data partition
    x_train, x_test,y_train, y_test = get_data(client_id)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    
       
    fl.client.start_numpy_client(
        server_address="127.0.0.1:5052",
        client=client,
    )



if __name__ == "__main__":
    main()

