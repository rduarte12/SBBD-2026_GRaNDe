import numpy as np
import config
from data_loader import load_data
from utils import create_folds, run_ball_tree
from gcn_model import GCNClassifier
import random
import torch
import tqdm
import sys
import pandas as pd

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    features_extractors= ['resnet'] 
    DATASET_TO_USE= 'flowers'
    GCN_TYPE_TO_USE= 'sgc_grande' # appnp_grande ou gcn_degree, sgc_grande ou sgc_degree
    NEURONS_LIST= [256]

    for FEATURES_EXTRACTOR in tqdm.tqdm(features_extractors, desc="Extratores de Features"):
        print(f'--- Iniciando Experimento para {DATASET_TO_USE.upper()} com features {FEATURES_EXTRACTOR.upper()} ---')

        features, labels= load_data(DATASET_TO_USE, FEATURES_EXTRACTOR)

        ranked_lists= run_ball_tree(features, k=config.L)
 
        folds= create_folds(features, labels, n_folds= config.N_folds)

        for neurons in tqdm.tqdm(NEURONS_LIST, desc="Número de Neurônios", leave=False):
            print(f"--- Treinando GCN com {neurons} neurônios na camada oculta ---")
            clf= GCNClassifier(GCN_TYPE_TO_USE, ranked_lists, len(labels), number_neighbors=config.k_graph, numbers_of_neurons=neurons)
            fold_acuracies= []

            """ for fold_idx, (test_idx, train_idx) in enumerate(folds):
                repetition_accuracies= []
                for _ in range(config.N_repetitions):
                    clf.prepare(test_index=test_idx, train_index=train_idx, features=features, labels=labels, matrix=correlation_matrix, limiar=threshold_to_use, correlation='default')
                    _, predictions= clf.train_and_predict()
                    test_labels= [labels[i] for i in test_idx]
                    acc= sum(1 for i, p in enumerate(predictions) if test_labels[i] == p) / len(predictions)
                    repetition_accuracies.append(acc)

                mean_acc_for_fold= np.mean(repetition_accuracies)
                fold_acuracies.append(mean_acc_for_fold) """
            
            list_mean_acuracies_per_repetition= []
            for _ in tqdm.trange(config.N_repetitions, desc="Repetições"):
                list_accuracies_per_fold= []
                for fold_idx, (test_idx, train_idx) in tqdm.tqdm(enumerate(folds), total=len(folds), desc="Folds", leave=False):
                    clf.prepare(test_index=test_idx, train_index=train_idx, features=features, labels=labels)
                    _, predictions= clf.train_and_predict()
                    test_labels= [labels[i] for i in test_idx]
                    acc= sum(1 for i, p in enumerate(predictions) if test_labels[i] == p) / len(predictions)
                    list_accuracies_per_fold.append(acc)
                mean_acc_for_repetition= np.mean(list_accuracies_per_fold)
                list_mean_acuracies_per_repetition.append(mean_acc_for_repetition)
            
            mean_accuracy= np.mean(list_mean_acuracies_per_repetition)
            std_accuracy= np.std(list_mean_acuracies_per_repetition, ddof=1)

            
            print("\n--- Resultados Finais ---")
            print(f"Dataset: {DATASET_TO_USE}, Features: {FEATURES_EXTRACTOR}")
            print(f"Acurácia Final: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")

    print("\n--- Experimento Concluído ---\n")

if __name__ == '__main__':
    main()