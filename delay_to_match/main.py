# from lightning_networks import dDMTSNet
from lightning_networks_swap import dDMTSNet
from lightning_task import dDMTSDataModule
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import argparse
from pytorch_lightning import Trainer
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy

import os

input_size = 8 + 3
#hidden_size = 100
output_size = 8 + 3
dt_ann = 15
alpha = dt_ann / 100
alpha_W = dt_ann / 100
g = 0.9

AVAIL_GPUS = max(1, torch.cuda.device_count())

BATCH_SIZE = 256 if AVAIL_GPUS else 32
# BATCH_SIZE = 1

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print('The number of availible GPUS is: ' + str(AVAIL_GPUS))

    parser = argparse.ArgumentParser(description="rnn dDMTS")

    # network parameters & hyperparameters
    parser.add_argument("--rnn_type", type=str,
                        default="vRNN", help="rNN to use")
    parser.add_argument("--nl", type=str, default="tanh",
                        help="nonlinearity to use")
    parser.add_argument("--hs", type=int, default="100", help="hidden size")
    parser.add_argument("--gamma", type=float, default=".005",
                        help="leak rate of anti-hebbian plasticity")

    # learning hyperparameters
    parser.add_argument("--lr", type=float, default="1e-3",
                        help="learning rate to use")
    parser.add_argument("--act_reg", type=float, default="1e-3",
                        help="activity regularization strength")
    parser.add_argument("--param_reg", type=float, default="1e-4",
                        help="parameter regularization strength")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train (default: 10)")
    
    parser.add_argument("--testing", action="store_true", help="Skip to testing if set")
    parser.add_argument("--plot", action="store_true", help="whether to plot or not")
    parser.add_argument("--include_delay", action="store_true", help="Whether to plot delay")
    parser.add_argument("--hemisphere", type=str, default=None,
                        help="left, right, or both hemisphere to be activated at initialization")
    parser.add_argument("--mode", type=str, default=None,
                        help="no-swap or swap mode to be activated at initialization")

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="_lightning_sandbox/checkpoints/",
        filename="rnn-sample-dDMTS-{epoch:02d}-{val_acc:.2f}--"
        + args.rnn_type
        + "--"
        + args.nl,
        every_n_epochs=args.epochs,
        mode="max",
        save_last=True,
    )

    '''
    checkpoint_callback.CHECKPOINT_NAME_LAST = ("last_rnn-sample-dDMTS-{epoch:02d}-{val_acc:.2f}--" + args.rnn_type    + "--"
    + args.nl
    )
    '''

    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"new-rnn={args.rnn_type}--nl={args.nl}--hs={args.hs}--act_reg={args.act_reg}--gamma={args.gamma}--param_reg={args.param_reg}--" + "{epoch:02d}--{val_acc:.2f}")

    early_stop_callback = EarlyStopping(
        monitor="val_acc", stopping_threshold=0.95, mode="max", patience=50
    )
    torch.set_float32_matmul_precision('medium') 
    
    # If the --testing flag is set, load the model from a checkpoint
    if args.testing:
        model = dDMTSNet(
            args.rnn_type,
            input_size,
            args.hs,
            output_size,
            dt_ann,
            alpha,
            alpha_W,
            g,
            args.nl,
            args.lr,
            args.include_delay,
            args.plot,
            args.hemisphere,
            args.mode
        )
        model = dDMTSNet.load_from_checkpoint("example.ckpt")
        print('model loaded from checkpoint')
    else:
        model = dDMTSNet(
            args.rnn_type,
            input_size,
            args.hs,
            output_size,
            dt_ann,
            alpha,
            alpha_W,
            g,
            args.nl,
            args.lr,
            args.include_delay,
            args.plot,
            args.hemisphere,
            args.mode
        )
        print('model initiated')

    model.act_reg = args.act_reg
    model.param_reg = args.param_reg

    if args.rnn_type == 'ah':
        model.rnn.gamma_val = args.gamma

    dDMTS = dDMTSDataModule(dt_ann=dt_ann)
    print('data initiated')

    tqdm_progress_bar = TQDMProgressBar()
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",
        enable_progress_bar=True,
        enable_model_summary=True
    )

    if not args.testing:
        trainer.fit(model, dDMTS)
        trainer.save_checkpoint("both_hemisphere_stsp.ckpt")
        print('training done.')
    
    trainer.test(model=model, datamodule=dDMTS)
    
    
    
