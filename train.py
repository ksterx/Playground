import argparse

from deepx.trainers import (
    ClassificationTrainer,
    ImageGenTrainer,
    LangModelTrainer,
    SegmentationTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algo", type=str, required=True)
parser.add_argument("-m", "--model", type=str, nargs="*", required=True)
parser.add_argument("-bb", "--backbone", type=str, default="resnet18")
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-do", "--dropout", type=float, default=0.0)
parser.add_argument("-r", "--root_dir", type=str, default="/workspace")
parser.add_argument("-mo", "--monitor", type=str, default="val_loss")
parser.add_argument("--monitor_max", action="store_true")
parser.add_argument("-p", "--patience", type=int, default=10)
parser.add_argument("-w", "--num_workers", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-lf", "--loss_fn", type=str, default="ce")
parser.add_argument("-o", "--optimizer", type=str, default="adam")
parser.add_argument("-s", "--scheduler", type=str, default="cos")
parser.add_argument("-ck", "--checkpoint", type=str, default=None)
parser.add_argument("-bm", "--benchmark", action="store_true")
parser.add_argument("-ratio", "--train_ratio", type=float, default=0.9)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--download", action="store_true")
args = parser.parse_args()


match args.algo:
    case "classification":
        trainer_cls = ClassificationTrainer
    case "segmentation":
        trainer_cls = SegmentationTrainer
    case "imagegen":
        trainer_cls = ImageGenTrainer
    case "langmodel":
        trainer_cls = LangModelTrainer
    case _:
        raise ValueError(f"Algorithm {args.algo} not supported")

if isinstance(args.model, str):
    args.model = [args.model]

for model in args.model:
    trainer = trainer_cls(
        model=model,
        datamodule=args.dataset,
        backbone=args.backbone,
        batch_size=args.batch_size,
        download=args.download,
        root_dir=args.root_dir,
        dropout=args.dropout,
        num_workers=args.num_workers,
        lr=args.learning_rate,
        loss_fn=args.loss_fn,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        train_ratio=args.train_ratio,
    )
    trainer.train(
        epochs=args.epochs,
        debug=args.debug,
        monitor=args.monitor,
        monitor_max=args.monitor_max,
        stopping_patience=args.patience,
        benchmark=args.benchmark,
        ckpt_path=args.checkpoint,
    )
