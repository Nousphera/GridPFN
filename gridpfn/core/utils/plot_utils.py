import logging
import os

import matplotlib.pyplot as plt

from gridpfn.core.training_config import MODEL_ORDER

logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
CUSTOM_NAMES = {
    "fedavg": "FedAvg",
    "edgehem": "EdgeHEM",
    "fedfit": "FedFit",
    "pffdst": "PFFDST",
    "feddmpq": "FedDMPQ",
}
MODEL_COLORS = {
    "fedavg": "#349EF5",
    "edgehem": "red",
    "fedfit": "#00A087",
    "pffdst": "#703D36",
    "feddmpq": "purple",
}


def save_figure_pdf(path, fig=None, dpi=300):
    base, ext = os.path.splitext(path)
    save_path = f"{base}.pdf" if ext.lower() != ".pdf" else path
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    return save_path


def order_models(model_names):
    ordered = [model for model in MODEL_ORDER if model in model_names]
    return ordered + [model for model in model_names if model not in MODEL_ORDER]
