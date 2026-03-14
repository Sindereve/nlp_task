import json
import os
from .glob_val import DATA_DIR, TARGET_DIR
import pandas as pd
from pandas import Series, DataFrame

def all_processing(
        y_data: list,
        categories: list
):
    for cat in categories:
        print('🏁 Начало кодировки.')
        index_in_class_dict, class_in_index_dict = get_dict_label_coder(pd.concat(y[cat] for y in y_data))
        
        target_dict_save(
            class_in_index_dict,
            index_in_class_dict,
            name=cat,
            data_dir=TARGET_DIR
        )

        for y in y_data :
            y[cat] = y[cat].apply(lambda x: class_in_index_dict[x])
        print('🍋‍🟩 Таргеты закодированны.')

def target_dict_save(
        class_in_index_dict: dict, 
        index_in_class_dict: dict, 
        name: str = 'base',
        data_dir: str = 'data/target'
):
    os.makedirs(data_dir, exist_ok=True)
    file_path = f"{data_dir}/{name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_to_class": index_in_class_dict,
            "class_to_index": class_in_index_dict
        }, f, ensure_ascii=False, indent=2)
    print(f"🟩 Словарь таргетов сохранён. Path:{file_path}")

def target_dict_load(
        name
):
    """
        Return:
            index_in_class_dict, class_in_index_dict
    """
    file_path = f"{TARGET_DIR}/{name}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)
    
    index_in_class_dict = {int(k): v for k, v in mappings["index_to_class"].items()}
    
    class_in_index_dict = mappings["class_to_index"]
    class_in_index_dict = {k: int(v) for k, v in class_in_index_dict.items()}
    
    print(f"🟩 Словари таргетов загружены. Path:{file_path}")
    return index_in_class_dict, class_in_index_dict



def get_dict_label_coder(
        serial: Series
):
    serial_unique = serial.unique()

    index_in_class_dict = {
        target_int:target_str for target_int, target_str in enumerate(serial_unique)
    }

    class_in_index_dict = {
        target_str:target_int for target_int, target_str in enumerate(serial_unique)
    }
    
    return index_in_class_dict, class_in_index_dict

def get_info(
        serial: pd.Series,
        top_n: int = 30,
        tail_n: int = 30,
):
    """
        Выводит статистику по распределению значений в Series.
    """
    
    counts = serial.value_counts().sort_values(ascending=False)
    n_unique = len(counts)
    
    print(f"Всего уникальных значений: {n_unique}")
    print(f"Всего записей: {len(serial):,}")
    
    if n_unique > (top_n + tail_n):
        top = counts.head(top_n)
        print(f"\nТоп-{top_n} классов:")
        print(f"  Сумма:        {top.sum():,}")
        print(f"  Среднее:      {top.mean():.1f}")
        print(f"  Мин / Макс:   {top.min()} / {top.max()}")
        
        tail = counts.tail(tail_n)
        print(f"\nХвост (последние {tail_n} классов):")
        print(f"  Сумма:        {tail.sum():,}")
        print(f"  Среднее:      {tail.mean():.1f}")
        print(f"  Мин / Макс:   {tail.min()} / {tail.max()}")
        
        total = counts.sum()
        print(f"\nДоля топ-{top_n}:  {top.sum() / total * 100:6.1f}%")
        print(f"Доля хвоста (последние {tail_n}): {tail.sum() / total * 100:6.1f}%")
        
        # Дополнительно: сколько классов покрывают 90% данных
        cum = counts.cumsum() / total
        if (cum >= 0.90).any():
            pos = (cum >= 0.90).argmax()
            n_for_90 = pos + 1
        else:
            n_for_90 = n_unique

        print(f"Для покрытия ≥90% данных нужно: {n_for_90} классов")
    
    else:
        print("\nМало уникальных классов — полное распределение:")
        print(counts)