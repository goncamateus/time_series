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


def plot_taylor(refs: dict, predictions_dict: dict):

    models = list(predictions_dict.keys())
    colors = ["c", "m", "y", "k", "r", "b", "g"]
    colors = colors[: len(models)]
    models = {model: color for model, color in zip(models, colors)}
    for idx, (model, pred_dict) in enumerate(predictions_dict.items()):
        taylor_stats = []
        name = model[0]
        if model.endswith("ND"):
            name = name + "ND"
        for horizon, pred in pred_dict.items():
            taylor_stats.append(
                sm.taylor_statistics(pred, refs[name][int(horizon)], "data")
            )

        sdev = np.array(
            [taylor_stats[0]["sdev"][0]] + [x["sdev"][1] for x in taylor_stats]
        )
        crmsd = np.array(
            [taylor_stats[0]["crmsd"][0]] + [x["crmsd"][1] for x in taylor_stats]
        )
        ccoef = np.array(
            [taylor_stats[0]["ccoef"][0]] + [x["ccoef"][1] for x in taylor_stats]
        )

        # To change other params in the plot, check SkillMetrics documentation in
        # https://github.com/PeterRochford/SkillMetrics/wiki/Target-Diagram-Options
        if len(list(predictions_dict.keys())) != 1:
            if (
                idx != len(list(predictions_dict.keys())) - 1
                or len(list(predictions_dict.keys())) == 1
            ):
                sm.taylor_diagram(
                    sdev,
                    crmsd,
                    ccoef,
                    styleOBS="-",
                    colOBS="g",
                    markerobs="o",
                    titleOBS="Observation",
                    markercolor=models[model],
                )
            else:
                sm.taylor_diagram(
                    sdev,
                    crmsd,
                    ccoef,
                    styleOBS="-",
                    titleOBS="Observation",
                    colOBS="g",
                    markerobs="o",
                    markercolor=models[model],
                    overlay="on",
                    markerLabel=models,
                )
        else:
            sm.taylor_diagram(
                sdev,
                crmsd,
                ccoef,
                styleOBS="-",
                colOBS="g",
                markerobs="o",
                titleOBS="Observation",
                markercolor="c",
                markerLabel=["placeholder"] + [k + 1 for k, v in pred_dict.items()],
            )


complexos = {
    "Amontada": ["BC", "IG", "RIB"],
    "Caldeirao": [
        "Santa_Angelina",
        "Santa_Barbara",
        "Santa_Ediwiges",
        "Santa_Fatima",
        "Santa_Regina",
        "Santo_Adriano",
        "Santo_Albano",
    ],
    "Icarai": ["Icarai_I", "Icarai_II"],
    "Riachao": [
        "Riachao_I",
        "Riachao_II",
        "Riachao_III",
        "Riachao_IV",
        "Riachao_V",
        "Riachao_VI",
        "Riachao_VII",
    ],
    "Taiba": ["Aguia", "Andorinha", "Colonia"],
}
steps = ["30_min"]
meses = ["Sep", "Oct", "Nov", "Dec"]
COMPLEXO_EOLICO = "Amontada"
results = {}
for COMPLEXO_EOLICO, centrais in complexos.items():
    for CENTRAL_EOLICA in centrais:
        for TIME_STEP in steps:
            for MES in meses:
                print()
                print()
                print()
                print(f'{COMPLEXO_EOLICO}->{CENTRAL_EOLICA}->{TIME_STEP}->{MES}')
                DECOMP = True
                DEVICE = torch.device("cuda")
                if not os.path.exists(
                    f"data/components/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}"
                ):
                    os.system(
                        f"mkdir data/components/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}"
                    )
                if not os.path.exists(f"data/out/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}"):
                    os.system(f"mkdir data/out/{CENTRAL_EOLICA}_{MES}_{TIME_STEP}")

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
                for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    again = True
                    while again:
                        print(
                            "--------------------------------------------------------"
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
                        trainer = Trainer(gpus=1, max_epochs=10)
                        trainer.fit(mlp, train_dataloaders=train_loader_lucas)
                        dataset_lucas.set_type("test")
                        mlp = mlp.cpu()
                        X_test_lucas = dataset_lucas.samples
                        y_mlp = (
                            mlp(X_test_lucas).detach()
                            / dataset_lucas.test_scaler.scale_
                        )
                        y_mlp = y_mlp.numpy()
                        latest_loss = trainer.callback_metrics['train_loss'].item() 
                        again = latest_loss < 0.022
                        if not again:
                            res = [None] * 3 + [None] * i + y_mlp.tolist()
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

for complexo, centrais in results.items():
    for central, times in centrais.items():
        for time, meses in times.items():
            for mes, horizons in meses.items():
                np.stack(horizons, axis=0).tofile(
                    f"data/out/{central}_{mes}_{time}.csv", sep=","
                )


