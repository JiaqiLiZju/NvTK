'''
@article{liaw2018tune,
    title={Tune: A Research Platform for Distributed Model Selection and Training},
    author={Liaw, Richard and Liang, Eric and Nishihara, Robert
            and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
    journal={arXiv preprint arXiv:1807.05118},
    year={2018}
}
'''
from torch.optim import Adam

from ray import tune

from . import Model
from .Trainer import Trainer

def objective(module, model_args):
    model = module(model_args)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    trainer = Trainer(model, criterion, optimizer, device)
    batch_losses, average_loss = trainer.train_per_epoch(train_loader, epoch, verbose_step=5)
    return average_loss


def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.grid_search([0.001, 0.01, 0.1]),
        "beta": tune.choice([1, 2, 3])
    })

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df