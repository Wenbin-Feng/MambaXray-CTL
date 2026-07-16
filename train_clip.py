import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.MambaXrayCTL_CLIP import MambaXrayCTLCLIP
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        max_epochs = args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    model = MambaXrayCTLCLIP(args)

    trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(args.seed, workers=True)
    train(args)


if __name__ == '__main__':
    main()
