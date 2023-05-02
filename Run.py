from Model.Dataset import ASAPDataset, loadData
from Model.Trainer import Trainer

from Model.Setting import TrainingSetting

MODEL_NAME: str = "my-model"

print("Model training begin...", flush = True)

train, validation, _ = loadData(ASAPDataset())
model = Trainer(MODEL_NAME)

for epoch in range(TrainingSetting.EPOCH):
    model.setMode(Trainer.OperationMode.TRAIN)
    model.train(train)
    
    model.setMode(Trainer.OperationMode.INFERENCE)
    model.validate(validation)

    model.advanceEpoch()

    model.checkpoint(MODEL_NAME + "-epoch_" + str(epoch))
    
print("Model training end", flush = True)