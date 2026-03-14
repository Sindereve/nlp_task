import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from tqdm import tqdm
import numpy as np
from numpy import ndarray 
from fasttext.FastText import _FastText

from .target_proc import target_dict_load
from . import glob_val

def file_save(
        sentences: ndarray,
        labels: ndarray,
        name:str = "_ft"
):
    with open(f"{glob_val.FT_DIR}/{name}.txt", "w") as f:
        for num, sentence in enumerate(sentences):
            label = str(labels[num])
            f.write(f"__label__{label} {' '.join(sentence)}\n")

def predict(
        model: _FastText, 
        text: str,
        dict_decoder: dict|None = None,
):
    """
    Return:
        label, score
    """
    result = model.predict(text)
    score = result[1][0]
    if dict_decoder:
        index = int(result[0][0].replace('__label__', ''))
        label = dict_decoder[index]
    else:
        label = int(result[0][0].replace('__label__', ''))

    return label, score

def cm_show(
        y_true,
        y_pred,
        classes,
        figsize = (18, 15)
):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_percent_rounded = np.round(cm_percent, 1)

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent_rounded,
        display_labels=classes
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap='Blues',
        values_format='.1f',
        colorbar=True
    )

    f1_sc = f1_score(y_true, y_pred, average='weighted')
    plt.title(f"Confusion Matrix в % ({f1_sc})")
    plt.xlabel("Pred class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()

def predict_and_cm_show(
        model,
        x_test,
        y_test,
        classes,
        figsize = (18, 15)
):
    """
    Reurn:
        list результаты предсказаний
    """
    y_pred = list()
    for x_ in tqdm(x_test):
        text_lem = " ".join(x_)
        y_pred.append(predict(model, text_lem)[0])

    cm_show(y_test, y_pred, classes, figsize)
    return y_pred 