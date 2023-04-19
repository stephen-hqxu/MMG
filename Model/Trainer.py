from Model.Component.Transformer import Transformer as Gen
from Model.Component.Discriminator import Discriminator as Disc

from Model.Setting import DatasetSetting, TrainingSetting

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import datetime
from enum import IntEnum

class Trainer():
    """
    Matching-based MIDI humanisation model training.

    @see https://github.com/soumith/ganhacks regarding choice of model architecture and hyperparameter.
    """

    class OperationMode(IntEnum):
        """
        @brief The mode of operation.
        """
        EVALUATION = 0x00
        TRAIN = 0xFF

    def __init__(this):
        """
        @brief Create a trainer with untrained model with random initial state.
        """
        this.Generator: Gen = Gen()
        this.Discriminator: Disc = Disc()

        # TODO: may want to use dynamic learning rate
        opt_param = { "lr" : TrainingSetting.LEARNING_RATE, "betas" : (TrainingSetting.BETA[0], TrainingSetting.BETA[1]) }
        this.GeneratorOptimiser: Adam = Adam(this.Generator.parameters(), **opt_param)
        this.DiscriminatorOptimiser: Adam = Adam(this.Discriminator.parameters(), **opt_param)

        # parameters to be updated during training
        this.Epoch: int = 0
        this.Loss: float = 0.0

    @classmethod
    def loadFrom(cls, model_name: str):
        """
        @brief Load a trainer from a saved model.

        @param model_name The name of the saved model.
        """
        # load saved data
        model = torch.load(DatasetSetting.MODEL_OUTPUT_PATH + '/' + model_name)
        trainer: cls = cls()

        # load each member data
        trainer.Generator.load_state_dict(model["generator"])
        trainer.Discriminator.load_state_dict(model["discriminator"])

        trainer.GeneratorOptimiser.load_state_dict(model["generator_optimiser"])
        trainer.DiscriminatorOptimiser.load_state_dict(model["discriminator_optimiser"])

        trainer.Epoch = model["epoch"]
        trainer.Loss = model["loss"]

        return trainer
    
    def checkpoint(this, model_name: str) -> None:
        """
        @brief Save the current state of the trainer to a file.

        @param module_name The name of the saving model.
        A datetime will be automatically appended to the end of the name.
        """
        time: str = str(datetime.datetime.today().strftime("%x_%X"))

        torch.save({
            "generator" : this.Generator.state_dict(),
            "discriminator" : this.Discriminator.state_dict(),

            "generator_optimiser" : this.GeneratorOptimiser.state_dict(),
            "discriminator_optimiser" : this.DiscriminatorOptimiser.state_dict(),

            "epoch" : this.Epoch,
            "loss" : this.Loss
            # filename extension follows PyTorch's convention
        }, DatasetSetting.MODEL_OUTPUT_PATH + '/' + model_name + '-' + time + ".tar")

    def setMode(this, mode: OperationMode) -> None:
        """
        @brief Set the model operation mode.

        @param mode The mode set to.
        """
        match(mode):
            case Trainer.OperationMode.EVALUATION:
                this.Generator.eval()
                this.Discriminator.eval()
            case Trainer.OperationMode.TRAIN:
                this.Generator.train()
                this.Discriminator.train()