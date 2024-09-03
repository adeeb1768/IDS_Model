import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import itertools
from typing import Dict, Optional, Tuple
import flwr as fl
import tensorflow as tf
from model_dataDiv import get_validation_data
import numpy as np
import matplotlib.pyplot as plt


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


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = my_model()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=21,
        min_evaluate_clients=21,
        min_available_clients=21,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:5052",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
        
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""


    # Use the validation set
    x_val, y_val = get_validation_data()

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
         # Calculate additional metrics from confusion matrix
        from sklearn.metrics import confusion_matrix

        y_pred = model.predict(x_val)  # Get model predictions on validation data
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels (assuming one-hot encoded labels)

        # Get confusion matrix
        cm = confusion_matrix(np.argmax(y_val,axis=1), y_pred_classes)

        # True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
        tn = cm[0, 0]
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]

        # Calculate FPR (False Positive Rate) and FNR (False Negative Rate)
        fpr = fp / (fp + tn)  # Ratio of misclassified normal traffic (FP) to total normal traffic
        fnr = fn / (fn + tp)  # Ratio of missed attacks (FN) to total actual attacks
        
        
        # Assuming you have class names (replace with your actual class names)
        class_names = ['DDoS-RSTFINFlood','DDoS-TCP_Flood','DDoS-ICMP_Flood','DoS-UDP_Flood','DoS-SYN_Flood','Mirai-greeth_flood',
                          'DDoS-SynonymousIP_Flood','Mirai-udpplain','DDoS-SYN_Flood','DDoS-PSHACK_Flood',
                          'DDoS-UDP_Flood','BenignTraffic']
        plot_confusion_matrix(cm, class_names)

        # Return loss, accuracy, and additional metrics
        return loss, {
            "accuracy": accuracy,
            "tn": tn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fpr": fpr,
            "fnr": fnr
            }    
       
    return evaluate


def fit_config(server_round: int):
    """
    Return training configuration dict for each round.
    Keep batch size fixed at 32, perform 4 rounds of training with 15 local epoch.
    """
    config = {
        "batch_size":32 , 
        "local_epochs": 15 
       
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

# Define function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
  """
  Plots a confusion matrix with labels.
  """
  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar(shrink=0.5)
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=90)
  plt.yticks(tick_marks, class_names)

  # Normalize confusion matrix values to percentages
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  # Print text with values inside the plot 
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)

  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
    main()
