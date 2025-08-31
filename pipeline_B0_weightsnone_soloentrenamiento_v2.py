#### este script es para correr en el drive
#%%
##import libraries
from tinyimagenet import TinyImageNet
from pathlib import Path
import torch
import torch.optim as optim
from torchvision import models
import torch.utils.data as data
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from torchvision import transforms as T

import poutyne

import pandas as pd
import os

import numpy as np
from pylab import *
import sys
import random

#/Users/tatibada/Documents/Tesis Maestria DM/scripts/EfficientNet/pipeline_B0_weightsnone_soloentrenamiento_v2.py
#sys.path.append('/content/drive/MyDrive/Tesis de Ms Data Mining TBadaracco/Notebooks/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
# load transformations
import transformaciones as tr

rotation_transforms = tr.rotation_transforms()
translation_transforms = tr.translation_transforms()
scale_transforms = tr.scale_transforms()
perspective_transforms = tr.perspective_transforms()
brightness_transforms = [tr.brightness_transforms(factor) for factor in tr.brightness_parameters]
contrast_transformations = [tr.contrast_transforms(alpha) for alpha in tr.contrast_list]
grayscale_transformations = [tr.grayscale_transforms(alpha) for alpha in tr.grey_list]
solarize_transformations = [tr.solarize_transforms(threshold) for threshold in tr.solarization_thresholds]
posterize_transformations = [tr.posterize_transforms(alpha) for alpha in tr.posterize_list]
invertion_transformations = [tr.invertion_transforms(alpha) for alpha in tr.invertion_list]

## escala de grises completar y rotacion correr completo

transformation_afin = [rotation_transforms,
                       translation_transforms,
                       scale_transforms,
                       perspective_transforms,
                       brightness_transforms,
                       contrast_transformations,
                       grayscale_transformations,
                       solarize_transformations,
                       posterize_transformations,
                       invertion_transformations]


transformaciones = ['rotacion','traslacion','escala','proyeccion','brillo','contraste','escala_grises','solarizacion','posterizacion','inversion_colores']

print('import transformations')

#### para salucionar error: RuntimeError: invalid hash value (expected "7eb33cd5", got "23ab8bcd5bdbef61a7a43b91adcad81f622fd7f36fb4935a569828d77888c44e")
#%%

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

#WeightsEnum.get_state_dict = get_state_dict
#####

# Definir el modelo y checkpoint
weights = None #models.EfficientNet_B0_Weights.IMAGENET1K_V1
base_model = models.efficientnet_b0(weights=weights)

for param in base_model.parameters():
    param.requires_grad = True

tinyimagenet_classes = 200
base_model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5, inplace=True),
    torch.nn.Linear(1280, tinyimagenet_classes),
)

model = torch.nn.Sequential(
   #T.Normalize(TinyImageNet.mean,TinyImageNet.std),
    #weights.transforms(),
    base_model,
)


model = model.to(device)
#%%
model
#%%

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Directorio principal donde se guardarán los resultados
main_dir = 'models/efficientnet_b0/weights_none'

# Crear directorio principal si no existe
Path(main_dir).mkdir(parents=True, exist_ok=True)

num_epochs = 50

class TinyImageNet(TinyImageNet):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x,y
#%%
# Iterar sobre cada transformación
for i, transformacion in enumerate(transformaciones[5:6]):
    print(f"Training with transformation: {transformacion}")

    # Crear directorio para la transformación actual
    transform_dir = os.path.join(main_dir, transformacion)
    Path(transform_dir).mkdir(parents=True, exist_ok=True)

    # Definir el path del checkpoint
    checkpoint_path = os.path.join(transform_dir, 'checkpoint_last.ckpt')
    checkpoint_opt_path = os.path.join(transform_dir, 'checkpoint_opt.ckpt') 

    print(f'checkpoint path: {checkpoint_path}')
    print(f'checkpoint optimizer path: {checkpoint_opt_path}')

    # Cargar o reiniciar el modelo y el checkpoint para cada transformación
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    # Carga el estado del optimizador desde el archivo
    if os.path.exists(checkpoint_opt_path):
        optimizer.load_state_dict(torch.load(checkpoint_opt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    # Definir el conjunto de datos con la transformación actual
    current_transform = transformation_afin[i]
    #print(current_transform)

    random_ts = lambda x: random.choice(current_transform)(x)
    
    normalize_transform = T.Compose(
        [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(TinyImageNet.mean,TinyImageNet.std),
        random_ts
        ])

    # Definir el conjunto de datos original
    dataset_train = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split="train", imagenet_idx=False, transform=normalize_transform)
    dataset_val = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split="val", imagenet_idx=False, transform=normalize_transform)

    #print(dataset_train[0])
    # Definir un DataLoader para cargar los datos originales por lotes
    train_loader = data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(dataset_val, batch_size=32, shuffle=True)


    print('DataLoader listo')


    # Definir el trainer
    trainer = poutyne.Model(
        model,
        optimizer,
        'cross_entropy',
        batch_metrics=['accuracy', poutyne.TopKAccuracy(5)],
        epoch_metrics=['f1'],
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Definir el historial y el checkpoint para cada transformación
    history_path = os.path.join(transform_dir, 'history.csv')
    checkpoint_path = os.path.join(transform_dir, 'checkpoint_last.ckpt')
    best_checkpoint_path = os.path.join(transform_dir, 'tmp_best.ckpt')
    checkpoint_opt_path = os.path.join(transform_dir, 'checkpoint_opt.ckpt')   

    checkpoint = poutyne.ModelCheckpoint(checkpoint_path, monitor='val_acc', mode='max', save_best_only=False, restore_best=False, verbose=True, temporary_filename=best_checkpoint_path)
    opt_checkpoint = poutyne.OptimizerCheckpoint(checkpoint_opt_path, monitor='val_acc', mode='max', save_best_only=False, restore_best=False, verbose=True,        temporary_filename=best_checkpoint_path)

    class HistorySaver(poutyne.Callback):

      def __init__(self,filepath):
        super().__init__()
        self.filepath = filepath
        self.history = []

      def on_epoch_end(self, epoch, logs):
        self.history.append(logs)

        if os.path.exists(history_path):
          df1 = pd.read_csv(history_path)
          df = pd.DataFrame(self.history)
          df = pd.concat([df1,df])
          df.reset_index(drop=True,inplace=True)
          df.drop_duplicates(inplace = True, ignore_index = True)
          df.to_csv(self.filepath,index=False)
        else:
          df = pd.DataFrame(self.history)
          df.to_csv(self.filepath,index=False)

    # callback personalizado
    history_saver = HistorySaver(history_path)

    if os.path.exists(history_path):
        df1 = pd.read_csv(history_path)
        history_saver.history = df1.to_dict('records')

    # Entrenar el modelo para cada transformación
    #history = trainer.fit_dataset(train_loader, valid_dataset=val_loader, batch_size=8, epochs=num_epochs, callbacks=[checkpoint, opt_checkpoint,history_saver])

    history = trainer.fit_generator(train_loader, val_loader, epochs=num_epochs, callbacks=[checkpoint, opt_checkpoint, history_saver])

# %%
