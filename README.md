# SPLN_Togedemaru_Semeval_Task_8

BioBERT

## Prerequisites

Install library dependencies:
```
python -m pip install -r requirements
```

Clone the following repositories:
```
git clone https://github.com/junwang4/bert-sklearn-with-class-weight
git clone https://github.com/junwang4/causal-language-use-in-science
```

Changes in 'causal-language-use-in-science' before training model:
```python
def get_class_weight(labels):
    class_weight = [x for x in compute_class_weight(class_weight="balanced",
                                                    classes=range(len(set(labels))),
                                                    y=labels)]
    print('- auto-computed class weight:', class_weight)
    return class_weight
```

```python
label_name = {0:'claim', 1:'per_exp', 2:'question', 3:'claim_per_exp'}

DATA_DIR = Path(__file__).parent.parent / 'datasets'

def get_train_data_csv_fpath(): return f'{DATA_DIR}/st1_train_inc_text_prep_out.csv'
```

## Run app with pretrained model
```
python main.py
```

