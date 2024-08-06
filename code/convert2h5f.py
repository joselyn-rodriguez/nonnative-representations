import numpy as np
import pandas as pd
import h5features as h5f
from tqdm import tqdm
import sys
import re
import math
import os 
print(pd.__version__)

def load_data(model, language):
   
    print(f'loading features for {model}')

    reps = np.load(f'/fs/clip-scratch/jrodri20/wav2vec/embeddings/{model}/{language}/{language}_reps.npy', allow_pickle=True)
    labels = np.load(f'/fs/clip-scratch/jrodri20/wav2vec/embeddings/{model}/{language}/{language}_labels.npy', allow_pickle=True)
    times = np.load(f'/fs/clip-scratch/jrodri20/wav2vec/embeddings/{model}/{language}/{language}_times.npy',allow_pickle=True)
    items = np.load(f'/fs/clip-scratch/jrodri20/wav2vec/embeddings/{model}/{language}/{language}_items.npy', allow_pickle=True)

    print('items loaded')
    return reps, labels, times, items


def run_convert2h5f(model, language):
    reps, labels, times, items = load_data(model, language)

    if 'wav2vec' in model:
        num_layers = 12
    if 'tera' in model:
        num_layers = 4


    out_dir = f"/fs/clip-scratch/jrodri20/wav2vec-h5features/{model}"
    feat_dir = f"{out_dir}/{language}"
    item_file = f'/fs/clip-scratch/jrodri20/abx/{language}.item'
    '''
    The reason why there's a difference for English and Hindi is because of an issue
    with each of the item files, for some reason one has file as index and one doesnt.
    '''
    print(f'converting {model} and {language} to h5f') # some small changes because the cluster is running on python 3.6 which has some old versions of pandas
    if 'eng' in language:
        print('here in eng')
        df = pd.read_table(item_file, index_col=0, header=0, delimiter=' ').reset_index().rename(columns={'index': 'file', '#file': 'onset', 'onset': 'offset', 'offset': 'phone', '#phone':'prev-phone', 'prev-phone': 'next-phone', 'next-phone':'speaker', 'speaker': 'extra'}) # might break again?
        df.drop(columns = 'extra')

    if 'hin' in language:
        df = pd.read_table(item_file, index_col=0, header=0, delimiter=' ').reset_index().rename(columns={'#file': 'file', '#phone':'phone'}) # might break again?

    print(df.head(5))

    for layer in range(0, num_layers):
        print(f'begining layer {layer}')
        writer = h5f.Writer(feat_dir+f'/{language}_features_{layer}.h5')
        for f in tqdm(set(df['file'].values)):
            indices = [i for i, x in enumerate(items) if x[:-4] == f] #f finding all indices for the files
            feat = [reps.item()[layer][i] for i in indices]
            feat = np.array(np.vstack(feat))
            labels = np.array([float(times[i]) for i in indices])
            post_data = h5f.Data([f], [labels], [feat], check=True) # h5f.Data(item_lst, label_lst, time_lst, check=True)
            writer.write(post_data, 'features', append=True) # you end up just appending all this data
        writer.close()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = "name of model to run")
    parser.add_argument('language', help = "language of representations")
    args = parser.parse_args()
    
    print('starting run convert2h5f')
    run_convert2h5f(args.model, args.language)