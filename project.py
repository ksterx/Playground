# %% [markdown]
# # プロジェクト

# %%
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, MetricCollection

# %%
# Colaboratoryでは以下を実行して必要なファイルをダウンロード
# !wget https://raw.githubusercontent.com/UTDataMining/2022A/master/project/winequality-white.csv

# %%
# 白ワインデータセットの読み込み
wine = pd.read_csv("winequality-white.csv", sep=";")

# %%
# 特徴量の分布
wine.hist(bins=50, figsize=(15, 15))

# %%
# 特徴量間の相関
wine.corr(method="pearson")

# %%
# 入力データ
X = wine[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
].values

# 正解ラベルを1（高品質：ワインのqualityが6以上）か0（低品質：ワインのqualityが6未満）とする
y = wine["quality"].values
y = (y >= 6).astype(int)
print(np.sum(y == 1, axis=0))  # 高品質ワインデータの数
print(np.sum(y == 0, axis=0))  # 低品質ワインデータの数

# %%
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# %%
# ロジスティック回帰モデル
lr = LogisticRegression(solver="liblinear", multi_class="auto")
lr.fit(X_train, y_train)  # 訓練データで学習
y_pred = lr.predict(X_test)  # テストデータで予測

print(confusion_matrix(y_test, y_pred))  # 混同行列
print(accuracy_score(y_test, y_pred))  # accuracy
print(precision_score(y_test, y_pred))  # precision
print(recall_score(y_test, y_pred))  # recall
print(f1_score(y_test, y_pred))  # f値

# %% [markdown]
# ## 課題
# **上記のワインの理化学検査結果からの品質予測について、あなた自身で（部下に対して模範となるような）分析に行い、その分析の過程と結果についてレポートとしてまとめたものを作成して提出してください。**
#
# - レポートは定められた様式に従って図表を含めて最大3ページまでとし、少なくとも1ページ以上は本文で構成してください（図表だけを貼り付けただけで説明がないものは不可）。
# - レポートと合わせて分析に用いたコードも提出してください。
#
# 授業で未習の内容を取り入れてもらっても構いませんが、それを持って大きく加点することはありません。むしろ、授業で学んだ内容を元にまず、適切に分析を行っているか、その上で独自の創意工夫があるか、を主に評価します。
#
# Conduct your own analysis (as an example to your colleague) of the wine quality prediction based on the chemical inspection results and summarize a report describing the process and results of your analysis.
# - The report should be no more than three pages long, including figures and tables, and at least one page should consist of a text part (the report with only figures and tables is not accepted).
# - Also submit the codes used in the analysis together with the report.
#
# You may use advanced topics that you have not yet learned in the class. However, it won't always lead to significant marks. The evaluation is rather based on whether you have first conducted relevant analysis based on what you have learned in the class, and whether you have any original and creative ideas on top of that.

# %%
class WineModel(pl.LightningModule):
    NUM_FEATURES = 11
    OUTPUT_SIZE = 2

    def __init__(self, hidden_size, lr, l2_lambda):
        super().__init__()
        self.lr = lr
        self.l2_lambda = l2_lambda

        self.loss_fn = nn.CrossEntropyLoss()
        metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # sequential model with batch normalization
        self.net = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.net(x.to(torch.float32))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_epoch_loss"}

    def training_step(self, batch, batch_index):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        self.train_metrics(pred, target)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return {
            "loss": loss,
            "target": target,
            "pred": pred,
        }

    def validation_step(self, batch, batch_index):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        self.val_metrics(pred, target)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(self.val_metrics, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return {
            "loss": loss,
            "target": target,
            "pred": pred,
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_epoch_loss", avg_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)


# %%
class WineData(pl.LightningDataModule):
    FEATURES = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    def __init__(self, batch_size, X_train, X_test, y_train, y_test, iqr_thresholds):
        super().__init__()

        self.batch_size = batch_size
        self.iqr_thresholds = iqr_thresholds

        # Preprocess data
        scaler = StandardScaler()
        X_train, y_train = self._remove_outliers(X_train, y_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train = torch.from_numpy(X_train)
        self.X_test = torch.from_numpy(X_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def _remove_outliers(self, x: np.ndarray, y):
        # (iqr_threshold) iqr or higher
        q1 = np.quantile(x, 0.25, axis=0)
        q3 = np.quantile(x, 0.75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.iqr_thresholds * iqr
        upper = q3 + self.iqr_thresholds * iqr
        mask = (x < lower) | (x > upper)
        mask = mask.any(axis=1)
        x = x[~mask]
        y = y[~mask]
        return x, y

    def plot(self):
        fig, ax = plt.subplots(4, 3, figsize=(8, 8))
        for i in range(len(self.FEATURES)):
            sns.histplot(self.X_train[:, i], ax=ax[i // 3, i % 3])
            sns.histplot(self.X_test[:, i], ax=ax[i // 3, i % 3])
            ax[i // 3, i % 3].set_title(f"{self.FEATURES[i]}")
        plt.show()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)


# %%
# Hyperparameters
epochs = 100
lr = 1e-4
hidden_size = 100
batch_size = 32
seed = 22

# %% [markdown]
#

# %%
def objective(trial):
    iqr_threshold = trial.suggest_float("iqr_threshold", 3, 5)
    iqr_thresholds = np.ones(11) * iqr_threshold
    iqr_thresholds[1] = trial.suggest_float("volatile acidity", 3, 5)
    iqr_thresholds[3] = np.inf
    iqr_thresholds[10] = trial.suggest_float("alcohol", 3, 5)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-6, 1e-4, log=True)

    torch.manual_seed(seed)
    data = WineData(
        batch_size=batch_size,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        iqr_thresholds=iqr_thresholds,
    )
    model = WineModel(hidden_size=hidden_size, lr=lr, l2_lambda=l2_lambda)
    logger = TensorBoardLogger("logs", name="wine")
    monitor = "val_MulticlassAccuracy_epoch"
    early_stopping = EarlyStopping(monitor=monitor, mode="max", patience=5)
    model_checkpoint = ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[early_stopping, model_checkpoint],
        gpus=0 if torch.cuda.is_available() else None,
    )
    trainer.fit(model, data)
    return trainer.checkpoint_callback.best_model_score.item()


# %%
# # Colab上だと時間がかかるのでコメントアウトしています
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

# %%
print(study.best_params)
print(study.best_value)

# %%
iqr_threshold = 3.0
iqr_thresholds = np.ones(11) * iqr_threshold
l2_lambda = 1e-05

torch.manual_seed(seed)
data = WineData(
    batch_size=batch_size, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, iqr_thresholds=iqr_thresholds
)
model = WineModel(hidden_size=hidden_size, lr=lr, l2_lambda=l2_lambda)
logger = TensorBoardLogger("logs", name="wine")
early_stopping = EarlyStopping(monitor="val_MulticlassAccuracy_epoch", mode="max", patience=5)
model_checkpoint = ModelCheckpoint(monitor="val_MulticlassAccuracy_epoch", mode="max", save_top_k=1, save_last=True)
trainer = pl.Trainer(max_epochs=epochs, logger=logger, callbacks=[early_stopping, model_checkpoint])

# %%
data.plot()

# %%
trainer.fit(model, data)

# %%
print(f"Accuracy: {trainer.checkpoint_callback.best_model_score.item()*100:.2f}%")

# %%
sns.pairplot(wine)

# %%
iqr_threshold = 3.89
l2_lambda = 3.77e-5

# %%
logger.metrics

# %%
