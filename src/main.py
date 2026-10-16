import numpy as np
import config
from data_loader import load_data
from utils import create_folds, run_ball_tree
from gcn_model import GCNClassifier
import random
import torch
import tqdm
import pandas as pd


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    NAME_EXPERIMENT = "name_experiment"

    features_extractors = ["resnet"]                 # resnet, senet, vit, dpnet, swimtf
    dATASET_TO_USE = ["flowers"]                     # flowers, cub200, dogs, pets, corel5k
    GCN_TYPE_TO_USE = "sgc_grande"                   # sgc_grande, sgc_degree, appnp_grande, appnp_degree
    GRAPH_TYPE_TO_USE = "rec"                        # knn or rec
    NEURONS_LIST = [256]                             # Is accept by APPNP. For SGC, it is fixed in
                                                     # gcn_model.py
    K_FOLDS_LIST = [10]                              # Number of folds (10 is recommended)
    K_NEIGHBORS_GRAPH = [40]                         # Number of neighbors for graph construction (max number of neighbors).

    # APPNP parameters
    APPNP_K_LIST = [10]                              # Number of propagation steps for APPNP (only for appnp_grande and appnp_degree).
    APPNP_ALPHA_LIST = [0.1]                         # Teleport probability for APPNP (only for appnp_grande and appnp_degree).
    
    # GRaNDe parameters 
    SIGMA = 0.2                                        # Sigma for GRaNDe (only for grande versions) 0.2 is recommended

    rows = []
    for DATASET_TO_USE in tqdm.tqdm(dATASET_TO_USE, desc="Datasets"):
        for FEATURES_EXTRACTOR in tqdm.tqdm(features_extractors, desc="Extratores de Features"):
            print(f"--- Iniciando Experimento para {DATASET_TO_USE.upper()} com features {FEATURES_EXTRACTOR.upper()} ---")

            features, labels = load_data(DATASET_TO_USE, FEATURES_EXTRACTOR)
            ranked_lists = run_ball_tree(features, k=config.L)

            for K_FOLDS in tqdm.tqdm(K_FOLDS_LIST, desc="Número de Folds", leave=False):
                folds = create_folds(features, labels, n_folds=K_FOLDS)

                for APPNP_K in tqdm.tqdm(APPNP_K_LIST, desc="K para APPNP", leave=False):
                    for APPNP_ALPHA in tqdm.tqdm(APPNP_ALPHA_LIST, desc="Alpha para APPNP", leave=False):
                    
                        for neurons in tqdm.tqdm(NEURONS_LIST, desc="Número de Neurônios", leave=False):

                            for k_neighborhood in tqdm.tqdm(K_NEIGHBORS_GRAPH, desc="K Vizinhos", leave=False):
                            
                                mean_acc_per_repetition = []
                                all_accuracies = []

                                for _ in tqdm.trange(config.N_repetitions, desc="Repetições", leave=False):
                                    acc_per_fold = []

                                    for fold_idx, (test_idx, train_idx) in tqdm.tqdm(
                                        enumerate(folds), total=len(folds), desc="Folds", leave=False
                                    ):
                                        if DATASET_TO_USE == "cub200":
                                            learning_rate = 0.01
                                        else:                                            
                                            learning_rate = config.learning_rate

                                        print(learning_rate)
                                            
                                        # number_neighbors agora recebe k_neighborhood 
                                        clf_final = GCNClassifier(gcn_type=GCN_TYPE_TO_USE,
                                                                  graph_type=GRAPH_TYPE_TO_USE,
                                                                  rks=ranked_lists,
                                                                  pN=len(labels),
                                                                  number_neighbors=k_neighborhood,
                                                                  numbers_of_neurons=neurons,
                                                                  learning_rate=learning_rate,
                                                                  numeroNeuronios=neurons,
                                                                  num_epochs=config.epochs, 
                                                                  sigma=SIGMA)

                                        clf_final.prepare(
                                            test_index=test_idx,
                                            train_index=train_idx, 
                                            features=features,
                                            labels=labels
                                        )

                                        _, predictions = clf_final.train_and_predict()
                                        test_labels = [labels[i] for i in test_idx]
                                        acc = sum(1 for i, p in enumerate(predictions) if test_labels[i] == p) / len(predictions)
                                        acc_per_fold.append(acc)
                                        all_accuracies.append(acc)


                                    # Salvando a média das repetições
                                    mean_acc_per_repetition.append(np.mean(acc_per_fold))
                                    print(f"Repetição concluída: Média de Acurácia = {mean_acc_per_repetition[-1]:.4f}")

                                mean_accuracy = np.mean(mean_acc_per_repetition)
                                std_accuracy_per_repetition = np.std(mean_acc_per_repetition, ddof=1)
                                std_accuracy_per_fold = np.std(all_accuracies, ddof=1)

                                if GCN_TYPE_TO_USE in ["appnp_grande", "appnp_degree"]:
                                    rows.append({
                                        "Dataset": DATASET_TO_USE,
                                        "Features": FEATURES_EXTRACTOR,
                                        "GCN Type": GCN_TYPE_TO_USE,
                                        "Neurons": neurons,
                                        "K_Folds": K_FOLDS,
                                        "APPNP_K": APPNP_K,
                                        "APPNP_ALPHA": APPNP_ALPHA,
                                        "SIGMA": SIGMA,
                                        "K_Neighborhood": k_neighborhood, # ALTERAÇÃO: Salvando o k_neighborhood atual
                                        "Mean Accuracy": mean_accuracy,
                                        "Std Accuracy Per Repetition": std_accuracy_per_repetition,
                                        "Std Accuracy Per Fold": std_accuracy_per_fold
                                    })
                                else:
                                    rows.append({
                                        "Dataset": DATASET_TO_USE,
                                        "Features": FEATURES_EXTRACTOR,
                                        "GCN Type": GCN_TYPE_TO_USE,
                                        "Neurons": neurons,
                                        "K_Folds": K_FOLDS,
                                        "APPNP_K": APPNP_K,
                                        "APPNP_ALPHA": APPNP_ALPHA,
                                        "SIGMA": SIGMA,
                                        "K_Neighborhood": k_neighborhood, # ALTERAÇÃO: Salvando o k_neighborhood atual
                                        "Mean Accuracy": mean_accuracy,
                                        "Std Accuracy Per Repetition": std_accuracy_per_repetition,
                                        "Std Accuracy Per Fold": std_accuracy_per_fold
                                    })

                                
                                print(f"\n -> Log: K_Neighborhood={k_neighborhood} | Mean Accuracy={mean_accuracy:.4f} | Std={std_accuracy_per_fold:.4f}")
                            

    df = pd.DataFrame(rows)
    df.to_csv(f"experiment_results_{GCN_TYPE_TO_USE}_{NAME_EXPERIMENT}.csv", index=False)
    print(df)
    print("\n--- Finished ---\n")

if __name__ == '__main__':
    main()
