# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import skill_metrics as sm
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.dataset import CERDataset
from src.mlp import MLPLucas

seaborn.set()

HORIZONS = 12
WINDOW = 3
DROPOUT = 0.0
DECOMP_METHOD = "2fft"

complexos = {
    "Amontada": ["BC", "IG", "RIB"],
    "Caldeirao": [
        "Santa_Angelina",
        "Santa_Barbara",
        "Santa_Edwiges",
        "Santa_Fatima",
        "Santa_Regina",
        "Santo_Adriano",
        "Santo_Albano",
    ],
    "Icarai": ["Icarai_I", "Icarai_II"],
    "Riachao": ["Riachao_I", "Riachao_II", "Riachao_IV", "Riachao_VI", "Riachao_VII",],
    "Taiba": ["Aguia", "Andorinha", "Colonia"],
}
steps = ["30_min", "1_day"]
meses = ["Sep", "Oct", "Nov", "Dec"]
results = {}
for COMPLEXO_EOLICO, centrais in complexos.items():
    for CENTRAL_EOLICA in centrais:
        for TIME_STEP in steps:
            for MES in meses:
                DECOMP = True
                DEVICE = torch.device("cuda")
                if not os.path.exists(
                    f"data/components/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}"
                ):
                    os.system(
                        f"mkdir data/components/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}"
                    )
                if not os.path.exists(f"data/out/{COMPLEXO_EOLICO}"):
                    os.system(f"mkdir data/out/{COMPLEXO_EOLICO}")

                dataset_lucas = CERDataset(
                    window=WINDOW,
                    horizons=HORIZONS,
                    complexo_eolico=COMPLEXO_EOLICO,
                    central_eolica=CENTRAL_EOLICA,
                    time_step=TIME_STEP,
                    mes=MES,
                    decomp=DECOMP,
                    decomp_method=DECOMP_METHOD,
                )
                for i in range(HORIZONS):
                    again = True
                    while again:
                        print(
                            "--------------------------------------------------------"
                        )
                        print()
                        print()
                        print()
                        print(
                            f"{COMPLEXO_EOLICO}->{CENTRAL_EOLICA}->{TIME_STEP}->{MES}"
                        )
                        print(f"Horizon: {i+1}")
                        dataset_lucas.set_type("train")
                        dataset_lucas.set_horizon(i)
                        train_loader_lucas = DataLoader(
                            dataset_lucas, batch_size=128, shuffle=True, num_workers=8
                        )
                        input_example_lucas = next(iter(train_loader_lucas))[0]
                        input_size_lucas = (
                            input_example_lucas.shape[1] * input_example_lucas.shape[2]
                        )

                        mlp = MLPLucas(
                            window_size=input_example_lucas.shape[1],
                            n_comps=input_example_lucas.shape[2],
                            horizons=1,
                        )
                        trainer = Trainer(gpus=1, max_epochs=30)
                        trainer.fit(mlp, train_dataloaders=train_loader_lucas)
                        dataset_lucas.set_type("test")
                        mlp = mlp.cpu()
                        X_test_lucas = dataset_lucas.samples
                        y_mlp = (
                            mlp(X_test_lucas).detach()
                            / dataset_lucas.test_scaler.scale_
                        )
                        y_mlp = y_mlp.numpy()
                        latest_loss = trainer.callback_metrics["train_loss"].item()
                        if TIME_STEP == "30_min":
                            again = latest_loss > 0.01
                        else:
                            again = False
                        if not again:
                            if TIME_STEP == "30_min":
                                res = [None] * 3 + [None] * i + y_mlp.tolist()
                            else:
                                ori_test_len = len(dataset_lucas.df_test)
                                res = y_mlp[-ori_test_len:]
                            res = np.array(res)
                            if COMPLEXO_EOLICO not in results:
                                results[COMPLEXO_EOLICO] = {}
                            if CENTRAL_EOLICA not in results[COMPLEXO_EOLICO]:
                                results[COMPLEXO_EOLICO][CENTRAL_EOLICA] = {}
                            if (
                                TIME_STEP
                                not in results[COMPLEXO_EOLICO][CENTRAL_EOLICA]
                            ):
                                results[COMPLEXO_EOLICO][CENTRAL_EOLICA][TIME_STEP] = {}
                            if (
                                MES
                                not in results[COMPLEXO_EOLICO][CENTRAL_EOLICA][
                                    TIME_STEP
                                ]
                            ):
                                results[COMPLEXO_EOLICO][CENTRAL_EOLICA][TIME_STEP][
                                    MES
                                ] = {}
                            results[COMPLEXO_EOLICO][CENTRAL_EOLICA][TIME_STEP][MES][
                                i + 1
                            ] = res
                res = pd.DataFrame(
                    results[COMPLEXO_EOLICO][CENTRAL_EOLICA][TIME_STEP][MES]
                )
                res = res.set_index(dataset_lucas.df_test["Timestamps"])
                res.to_csv(
                    f"data/out/{COMPLEXO_EOLICO}/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}.csv",
                    header=False,
                )
