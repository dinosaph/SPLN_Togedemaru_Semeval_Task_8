# ==================================================================================================================================
# IMPORTS
# ==================================================================================================================================

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
from pathlib import Path
from bert_sklearn import load_model

# ==================================================================================================================================
# GLOBALS
# ==================================================================================================================================

tqdm.pandas()

RE_EMAIL = re.compile(r'[A-Za-z0-9]{1,}@[A-Za-z]{1,}\.[A-Za-z0-9]{1,}')
RE_URL = re.compile(r'https?://\S+')
RE_HASH = re.compile(r'(#[A-Za-z0-9]+)')
RE_MEN = re.compile(r'(@[A-Za-z0-9]+)')
RE_SYM = re.compile(r"( -)|(- )|(\d+)|([`~!@#$%^&*()_+[\]{};:'\"\\|,<.>/]+)")

# ==================================================================================================================================
# CLASS - Causal Extractor
# ==================================================================================================================================

class CausalExtractor:
    
    def __init__(self) -> None:
        
        print(f"> Loading Causal Extractor model.")
        self.model = load_model(Path(__file__).parent / 'model' / 'EMNLP_biobert' / 'K5_epochs5_2.bin')
     
    @staticmethod
    def get_baseline(file_path: str):
        
        print(f"> Baseline alg. checks.")
        df = pd.read_csv(file_path)
        
        X = np.array(df['sentence'])
        y = np.array(df['label'])
        
        strategies = ['most_frequent', 'stratified', 'uniform', 'constant']
        # Splitting the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.3, random_state=0)
        
        test_scores = []
        for s in strategies:
            if s =='constant':
                dclf = DummyClassifier(strategy=s, random_state=0, constant=2)
            else:
                dclf = DummyClassifier(strategy=s, random_state=0)
            dclf.fit(X_train, y_train)
            score = dclf.score(X_test, y_test)
            test_scores.append(score)
        
        ax = sns.stripplot(x=strategies, y=test_scores)
        ax.set(xlabel='Strategy', ylabel='Test Score')
        plt.show()
        
    def predict(self, doc):
        
        return self.model.predict(doc)
    
    def get_predictions_st1(self, file_path: str):
        
        print(f"> Getting predictions for stage 1 using pretrained model.")
        df = pd.read_csv(file_path)
        df['labels'] = self.predict(np.array(df['text']))
        self.generate_submission_st1(df)
    
    def generate_submission_st1(self, df):
        
        print(f"> Generating output for stage 1 predictions.")
        df.insert(2, 'words', df['text'].str.split())
        df = df.explode('words')
        df = df.drop(['text'], axis=1)
        labels = list(PreprocessExpert.LABELS_TO_INT.keys())
        df['labels'] = [labels[l] for l in np.array(df['labels'])]
        new_file_path = Path(__file__).parent / 'st1_pred.csv'
        df.to_csv(new_file_path, index=False)

# ==================================================================================================================================
# CLASS - Preprocess Expert
# ==================================================================================================================================

class PreprocessExpert:
    
    LABELS_TO_INT = {
        'claim': 0,
        'per_exp': 1,
        'question': 2,
        'claim_per_exp': 3
    }
    
    @staticmethod
    def clean_text(doc: str):
        
        new_doc = doc.replace('\n', ' ').strip()
        new_doc = RE_EMAIL.sub("", new_doc)
        new_doc = RE_HASH.sub("", new_doc)
        new_doc = RE_MEN.sub("", new_doc)
        new_doc = RE_SYM.sub("", new_doc)
        new_doc = RE_URL.sub("", new_doc)
        
        return new_doc
    
    @staticmethod
    def get_prep_reddit_train_df(file_path: str):
        
        print(f"\nPreparing train data from Reddit.")
        df = pd.read_csv(file_path)
        sentence_n_label = []
        for i in range(len(df)):
            labeled_sections = eval(df.iloc[i]['stage1_labels'])[0]['crowd-entity-annotation']['entities']
            for j in range(len(labeled_sections)):
                s = df.iloc[i]['text'][:labeled_sections[j]['startOffset']].rfind(' ')
                e = df.iloc[i]['text'][:labeled_sections[j]['endOffset']].rfind(' ')
                sentences = re.split('[,.?!]', df.iloc[i]['text'][s:e])
                sentences = [st for st in sentences if st != '']
                if len(sentences) > 0:
                    for sentence in sentences:
                        if len(sentence.split(' ')) > 2:
                            sentence_n_label.append({
                                'sentence': sentence.strip(),
                                'label': PreprocessExpert.LABELS_TO_INT[labeled_sections[j]['label']]
                            })
        df_new = pd.DataFrame.from_records(sentence_n_label)
        df_new['sentence'] = df_new['sentence'].progress_apply(PreprocessExpert.clean_text)
        
        new_file_path = Path(file_path).parent / str(Path(file_path).stem + '_prep_out.csv')
        df_new.to_csv(new_file_path, index=False)
        return new_file_path
    
    @staticmethod
    def get_prep_reddit_test_df(file_path: str):
        
        print(f"\nPreparing test data from Reddit.")
        df = pd.read_csv(file_path)
        data_n_sentences = []
        for i in range(len(df)):
            current_post = df.iloc[i]
            post_sentences = re.split('[,.?!]', current_post['text'])
            post_sentences = [st for st in post_sentences if st != '']
            if len(post_sentences) > 0:
                for ps in post_sentences:
                    if len(ps.strip()) > 0 and len(ps.split(' ')) > 3:
                        data_n_sentences.append({
                            'post_id': current_post['post_id'],
                            'subreddit_id': current_post['subreddit_id'],
                            'text': ps.strip(),
                        })
        
        df_new = pd.DataFrame.from_records(data_n_sentences)
        df_new['text'] = df_new['text'].progress_apply(PreprocessExpert.clean_text)
        
        new_file_path = Path(file_path).parent / str(Path(file_path).stem + '_prep_out.csv')
        df_new.to_csv(new_file_path, index=False)
        return new_file_path

# ==================================================================================================================================
# RUN / TEST
# ==================================================================================================================================

if __name__ == '__main__':

    file_train = PreprocessExpert.get_prep_reddit_train_df(r'datasets/st1_train_inc_text.csv')
    file_test = PreprocessExpert.get_prep_reddit_test_df(r'datasets/st1_test_inc_text.csv')

    CausalExtractor.get_baseline(file_train)

    # ce = CausalExtractor()
    # ce.get_predictions_st1(file_test)