# app.py
from rfdetr import RFDETRBase
import os

# Initialize the model
model = RFDETRBase()
history = []

# Callback function to store training history
def callback2(data):
    history.append(data)

# Add callback to model's callback list
model.callbacks["on_fit_epoch_end"].append(callback2)

# Main block to prevent multiprocessing issues on Windows
if __name__ == '__main__':
    # Set environment variable to reduce GPU memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set the path to your local dataset (replace with your actual path)
    dataset_dir = "mahjong-vtacs-mexax-m4vyu-sjtd-2"  # Adjust this path
    
    # Verify if the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist. Please provide a valid path.")

    # Training parameters
    epochs = 5
    batch_size = 4
    grad_accum_steps = 1
    learning_rate = 1e-4

    # Start training
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=learning_rate
    )

    # Print training completion message
    print("Training completed. History length:", len(history))