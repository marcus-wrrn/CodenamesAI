import torch
import matplotlib.pyplot as plt

def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_plot(losses_train: list, losses_test: list, save_path: str):
    # Plot training losses
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')
    plt.plot([i for i in range(len(losses_test))], losses_test, label='Test Loss')
    # Set the title and labels
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()