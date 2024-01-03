import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_plot(losses_train: list, losses_test: list, save_path: str):
    # Plot training losses
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')
    plt.plot([i for i in range(len(losses_test))], losses_test, label='Test Loss')
    # Set labels
    #plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def calc_game_scores(model_out: Tensor, pos_encs: Tensor, neg_encs: Tensor, neut_encs: Tensor, device: torch.device):
    model_out_expanded = model_out.unsqueeze(1)
    pos_scores = F.cosine_similarity(model_out_expanded, pos_encs, dim=2)
    neg_scores = F.cosine_similarity(model_out_expanded, neg_encs, dim=2)
    neut_scores = F.cosine_similarity(model_out_expanded, neut_encs, dim=2)

    combined_scores = torch.cat((pos_scores, neg_scores, neut_scores), dim=1)
    _, indices = combined_scores.sort(dim=1, descending=True)
    # create reward copies
    pos_reward = torch.zeros(pos_scores.shape[1]).to(device)
    neg_reward = torch.ones(neg_scores.shape[1]).to(device) * 2
    neut_reward = torch.ones(neut_scores.shape[1]).to(device) 

    combined_rewards = torch.cat((pos_reward, neg_reward, neut_reward))
    combined_rewards = combined_rewards.expand((combined_scores.shape[0], combined_rewards.shape[0]))
    rewards = torch.gather(combined_rewards, 1, indices)

    non_zero_mask = torch.where(rewards != 0, 1., 0.)
    num_correct = torch.argmax(non_zero_mask, dim=1)
    first_incorrect_value = rewards[torch.arange(rewards.size(0)), num_correct]
    return num_correct.float().mean(), first_incorrect_value.mean()


