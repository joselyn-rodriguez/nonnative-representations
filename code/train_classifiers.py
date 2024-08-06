from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import json
import os
import random


"""
trains multiple different classifiers by-layer

input
    - language (to save files)
    - data_path, the directory where you want things to be saved 
    - data_X, all input features
    - data_Y, all input labels
"""

random_seed = 32

np.random.seed(random_seed)


def get_category_counts(data_Y):
    '''
    returns a dictionary of phones and counts in the form: 
    {'SIL': 21401, 'a': 2391, 'aː': 35719, ...}
    '''
    unique, counts = np.unique(data_Y, return_counts=True)
    categories = dict(zip(unique, counts))
    return categories
        

def subsampling_data(data_X, data_Y, encoder):
    random.seed = 42
    categories = get_category_counts(data_Y)
    
    least_freq_count = 67 # categories[encoder.transform(['ɖ̤'])[0]]
    
    all_indices = np.array([]).astype('int') 
    
    for i in np.unique(data_Y):
        # print(f'current i: {i}')
        indices = [index for index, label in enumerate(data_Y) if label == i]
        index_sample = random.sample(indices, min(categories[i],least_freq_count )) # trying to randomly sample for a given category
        all_indices = np.append(all_indices, index_sample)
        # print("length of index sample", len(index_sample))
   
    print(all_indices)
    
    data_X_ss = np.array([data_X[i] for i in all_indices])
    data_Y_ss = np.array([data_Y[i] for i in all_indices])
    
    return data_X_ss, data_Y_ss    


def train_strat_classifier(load_dir, language, data_path):   
    print("THIS IS THE LOADING DIRECTORY: ", load_dir)
    data_X = np.load(os.path.join(load_dir, f'{language}_reps.npy'), allow_pickle=True)
    data_Y = np.load(os.path.join(load_dir, f'{language}_labels.npy'), allow_pickle=True)
    data_path = os.path.join(data_path, language)
    
    # CREATE LABEL ENCODER
    encoder = LabelEncoder()
    encoder.fit(data_Y)
    encoded_data_Y = encoder.transform(data_Y)
    np.save(os.path.join(data_path, f'{language}_classes.npy'), encoder.classes_)


    data_X_uw = data_X.item() # if you're importing from numpy you need to unwrap the data
    num_layers=len(data_X_uw)
    
    accuracy = open(os.path.join(data_path, f"{language}_accuracy.txt"), 'x')
    accuracy.close() 
    accuracy = open(os.path.join(data_path,  f"{language}_accuracy.txt"), 'a')
    
    print(f'number of layers: {num_layers}')

    for layer in tqdm(range(0,  num_layers)):

        data_X_reshaped = np.vstack(data_X_uw[layer])
        
        data_X_subsampled, data_Y_subsampled = subsampling_data(data_X_reshaped, encoded_data_Y, encoder)
        
        '''
        using stratified K fold, you should be getting the same-ish number of categories in each split. 
        '''
        skf = StratifiedKFold(n_splits=5) 

        for strat, (train, test) in enumerate(skf.split(data_X_subsampled, data_Y_subsampled)):
#             print(f'train: {train} \n test: {test}')

            print('begin model training')
            clf_log_reg_sdg = SGDClassifier(loss='log_loss', alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, \
                                        shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', \
                                        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, \
                                        class_weight=None, warm_start=False, average=False).fit(data_X_subsampled[train], data_Y_subsampled[train])
            print('model training complete')

            X_pred = clf_log_reg_sdg.predict(data_X_subsampled[test])

            with open(os.path.join(data_path, f'sdg_classifier_{layer}_strat{strat}.pkl'), 'wb') as f:  # open a text file
                pickle.dump(clf_log_reg_sdg, f) # serialize the list

            acc_score = accuracy_score( data_Y_subsampled[test], X_pred)
            accuracy.writelines(f'{layer}_{strat}_' + str(acc_score) + '\n')

            full_report = metrics.classification_report(data_Y_subsampled[test], X_pred, output_dict=True)
            df_full_report = pd.DataFrame(full_report).transpose()
            full_data = pd.DataFrame(

                {'predicted_label': encoder.inverse_transform(X_pred),
                 'actual_label': data_Y_subsampled[test],
                })
    
            with open(os.path.join(data_path, f'X_test_data_{layer}_strat{strat}.pkl'), 'wb') as g:  # open a text file
                    pickle.dump(data_X_subsampled[test], g) # serialize the list

            with open(os.path.join(data_path, f'y_test_data_{layer}_strat{strat}.pkl'), 'wb') as h:  # open a text file
                    pickle.dump(data_Y_subsampled[test], h) # serialize the list

            df_full_report.to_pickle(os.path.join(data_path, f'{language}_report_{layer}_strat{strat}.pkl'))
            full_data.to_pickle(os.path.join(data_path, f'{language}_full_data_{layer}_strat{strat}.pkl'))

            print(f'data saved for layer {layer}')
    
    accuracy.close()



# fire wasn't working otherwise I would have used fire.Fire()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help = "the source file for the global phone directory (wav files)")
    parser.add_argument('language', help = "folder that has ur dictionary")
    parser.add_argument('data_path', help = "where do you want the output to go?")
    args = parser.parse_args()
    
    train_strat_classifier(args.load_dir, args.language, args.data_path)

