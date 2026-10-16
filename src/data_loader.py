import numpy as np

def load_data(dataset_name, feat_extractor):

    if dataset_name == 'flowers':
        if feat_extractor == 'resnet':
            data = np.load('./data/cnn-last_linear-resnet152.npz')
            features = data['features']
        else:  
            raise ValueError(f"Extrator de features '{feat_extractor}' não suportado para o dataset 'flowers'.")
        
        labels = []
        with open('./data/flowers_classes.txt', "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                _ , label = line.strip().split(":")
                labels.append(int(label))

        return np.array(features), np.array(labels)    

    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")
