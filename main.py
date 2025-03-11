from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_history

def main():
    print("Starting the facial recognition and emotion detection project...")

    # Step 1: Train the model
    print("Training the model...")
    history = train_model()  # train_model should return the history object

    # Step 2: Plot training history (optional)
    print("Plotting training history...")
    plot_history(history)

    # Step 3: Evaluate the model
    print("Evaluating the model on test data...")
    evaluate_model()

    print("Process completed.")

if __name__ == "__main__":
    main()
