#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Función para redondear columnas numéricas y eliminar duplicados
def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    df.drop_duplicates(inplace=True)
    print(df.shape)
    df.reset_index(inplace=True,drop=True)
    df['epoch'] = df.index
    df.to_csv(file_path)
    return df

# Función para graficar y guardar la imagen
def plot_and_save(df, save_path, x_label, y_label1, y_label2):
    plt.plot(df['epoch'], df[y_label1], label=y_label1)
    plt.plot(df['epoch'], df[y_label2], label=y_label2)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Valor')
    plt.title(f'Gráfico de {y_label1} y {y_label2} por época - {folder_name}')
    plt.savefig(save_path)
    plt.close()
#%%
# Directorio principal donde se encuentran las carpetas
main_directory =  '/home/tbadaracco/models/efficientnet_b0'
print(main_directory)
folder_path = os.path.join(main_directory, 'weights_none')
print(folder_path)
#%%
# Recorrer las carpetas
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            try:
                # Obtener el nombre de la carpeta actual
                folder_name = os.path.basename(root)
                print(folder_name)
                # Obtener el nombre del archivo sin la extensión
                file_name = os.path.splitext(file)[0]
            
                # Preprocesar el archivo CSV
                df = preprocess_csv(file_path)

                # Nombre de salida para los gráficos
                output_filename_acc = f'{file_name}_acc.png'
                output_filename_loss = f'{file_name}_loss.png'
                
                # Graficar y guardar la imagen para loss y val_loss
                plot_and_save(df, os.path.join(root, output_filename_loss), 'Epoch', 'loss', 'val_loss')
                
                # Graficar y guardar la imagen para acc y val_acc
                plot_and_save(df, os.path.join(root, output_filename_acc), 'Epoch', 'acc', 'val_acc')

                print(f'Se han generado los gráficos para {file}')
            except Exception as e:
                print(f'Error al procesar {file}: {e}')
# %%
## vuelvo a levantar y creo una tabla completa para calcular performance
df_complete = pd.DataFrame()
for root, dirs, files in os.walk(main_directory):
    #print(files)
    for file in files:
        if file.endswith(".csv") and 'checkpoint' not in file:
            file_path = os.path.join(root, file)

            # Leer el archivo CSV
            df = pd.read_csv(file_path)
            directorio_padre = os.path.dirname(file_path)
            nombre_directorio = os.path.basename(directorio_padre)
            
            ## agrego columnas entrenado
            df['Training'] = nombre_directorio
            ## concateno
            df_complete = pd.concat([df_complete,df])
#%%
df_complete = df_complete[['epoch', 'time', 'loss', 'acc', 'top5',
       'fscore_macro', 'val_loss', 'val_acc', 'val_top5', 'val_fscore_macro',
       'Training']]
#%%
df_complete.to_csv('complete_history.csv',index=False)
#%% lo sgte es para volver a correr solo para una transformacion
transf = 'contraste'
df1 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_final_training.csv')

print(df1.shape)
df1 = df1.loc[df1.Transformation_training != transf]
print(df1.shape)
df1.to_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_final_training.csv',index = False)
#df1 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_final.csv')
df1 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_progress_training.csv')

print(df1.shape)
df1 = df1.loc[df1.Transformation_training != transf]
print(df1.shape)

df1.to_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_progress_training.csv',index = False)


df1 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_progress.csv')
print(df1.shape)
df1 = df1.loc[df1.Transformation != transf]
print(df1.shape)
df1.to_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_progress.csv',index=False)
#%%
import os
import pandas as pd
from torchvision import models
from tinyimagenet import TinyImageNet
from torchvision import transforms as T
from pathlib import Path
import torch.utils.data as data
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm  # Importar tqdm para las barras de progreso

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
   base_model,
)

main_directory = '/home/tbadaracco/models/efficientnet_b0/weights_none'

normalize_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(TinyImageNet.mean, TinyImageNet.std),
    ]
)

dataset_train = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split="train", imagenet_idx=False, transform=normalize_transform)
dataset_val = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split="val", imagenet_idx=False, transform=normalize_transform)

train_loader = data.DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = data.DataLoader(dataset_val, batch_size=32, shuffle=True)

# Ruta para guardar el progreso
progress_file = os.path.join(main_directory, 'performance_metrics_progress.csv')

# Cargar progreso anterior si existe
if os.path.exists(progress_file):
    df_progress = pd.read_csv(progress_file)
    completed_transformations = set(df_progress['Transformation'])
else:
    df_progress = pd.DataFrame()
    completed_transformations = set()

results = []

for root, dirs, files in os.walk(main_directory):
    if '.ipynb_checkpoints' in dirs:
        dirs.remove('.ipynb_checkpoints')
    for dir in tqdm(dirs, desc="Evaluating models"):
        dir_path = os.path.join(root, dir)
        model_path = os.path.join(dir_path, 'checkpoint_last.ckpt')

        results_folder_original = os.path.join(root, 'Performance')
        directorio_padre = os.path.dirname(model_path)
        nombre_directorio = os.path.basename(directorio_padre)

        # Si ya se evaluó esta transformación, saltarla
        if nombre_directorio in completed_transformations:
            print(f"Skipping already evaluated transformation: {nombre_directorio}")
            continue

        print(f"\nEvaluating model from directory: {nombre_directorio}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint

        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        # Evaluación en el conjunto de entrenamiento
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []
        total_loss_train = 0.0

        print("Evaluating on training set...")
        with torch.no_grad():
            for data in tqdm(train_loader, desc="Training set", leave=False):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss_train += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                all_labels_train.extend(labels.cpu().numpy())
                all_predictions_train.extend(predicted.cpu().numpy())

        accuracy_train = correct_train / total_train
        average_loss_train = total_loss_train / len(train_loader)
        precision_train = precision_score(all_labels_train, all_predictions_train, average='weighted')
        recall_train = recall_score(all_labels_train, all_predictions_train, average='weighted')

        print(f"Training set - Accuracy: {accuracy_train:.4f}, Loss: {average_loss_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}")

        # Evaluación en el conjunto de validación
        correct_val = 0
        total_val = 0
        all_labels_val = []
        all_predictions_val = []
        total_loss_val = 0.0

        print("Evaluating on validation set...")
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation set", leave=False):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_labels_val.extend(labels.cpu().numpy())
                all_predictions_val.extend(predicted.cpu().numpy())

        accuracy_val = correct_val / total_val
        average_loss_val = total_loss_val / len(val_loader)
        precision_val = precision_score(all_labels_val, all_predictions_val, average='weighted')
        recall_val = recall_score(all_labels_val, all_predictions_val, average='weighted')

        print(f"Validation set - Accuracy: {accuracy_val:.4f}, Loss: {average_loss_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}")

        results.append({
            'Transformation': nombre_directorio,
            'Accuracy_Train': accuracy_train,
            'Loss_Train': average_loss_train,
            'Precision_Train': precision_train,
            'Recall_Train': recall_train,
            'Accuracy_Val': accuracy_val,
            'Loss_Val': average_loss_val,
            'Precision_Val': precision_val,
            'Recall_Val': recall_val
        })

        # Guardar el progreso después de cada iteración
        df_progress = pd.DataFrame(results)
        df_progress.to_csv(progress_file, index=False)

# Guardar resultados finales en un CSV separado
df_final = pd.DataFrame(results)
df_final.to_csv(os.path.join(main_directory, 'performance_metrics_final.csv'), index=False)

print("Resultados finales guardados en performance_metrics_final.csv")

# %% Preparo la tabla para calcular el headmap con el accuracy de training por cada trasnformacion y evaluando en las distintas transformaciones
import os
import pandas as pd
from torchvision import models
from tinyimagenet import TinyImageNet
from torchvision import transforms as T
from pathlib import Path
import torch.utils.data as data
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm  # Importar tqdm para las barras de progreso
import random
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
#%%
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
   base_model,
)

main_directory = '/home/tbadaracco/models/efficientnet_b0/weights_none'
#%%

# Ruta para guardar el progreso
#progress_file = os.path.join(main_directory, 'performance_metrics_progress_training.csv')
progress_file = os.path.join(main_directory, 'performance_metrics_final_training.csv')

# Cargar progreso anterior si existe
if os.path.exists(progress_file):
    df_progress1 = pd.read_csv(progress_file)
    completed_transformations = set(df_progress1['Transformation_training'])
    completed_transformations = set(zip(df_progress1['Transformation_training'], df_progress1['Transformation_eval']))

else:
    df_progress = pd.DataFrame()
    completed_transformations = set()

results = []

for root, dirs, files in os.walk(main_directory):
    if '.ipynb_checkpoints' in dirs:
        dirs.remove('.ipynb_checkpoints')
    for dir in tqdm(dirs, desc="Evaluating models"):
        dir_path = os.path.join(root, dir)
        model_path = os.path.join(dir_path, 'checkpoint_last.ckpt')

        results_folder_original = os.path.join(root, 'Performance')
        directorio_padre = os.path.dirname(model_path)
        nombre_directorio = os.path.basename(directorio_padre)

        # Si ya se evaluó esta transformación, saltarla
        #if nombre_directorio in completed_transformations:
        #    print(f"Skipping already evaluated transformation: {nombre_directorio}")
        #    continue

        print(f"\nEvaluating model from directory: {nombre_directorio}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint

        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        # Evaluación en el conjunto de entrenamiento
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []
        total_loss_train = 0.0

        print("Evaluating on training set...")
        with torch.no_grad():
            ## lo hago para cada trasnformacion 
            for i, transformacion in enumerate(transformaciones):
                print(f"evaluating with transformation: {transformacion}")
                # Verificar si el par de transformaciones ya fue evaluado
                if (nombre_directorio, transformacion) in completed_transformations:
                    print(f"Skipping already evaluated pair: {nombre_directorio}, {transformacion}")
                    continue
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

                dataset_train = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), split="train", imagenet_idx=False, transform=normalize_transform)

                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)


                for data in tqdm(train_loader, desc="Training set", leave=False):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss_train += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    
                    all_labels_train.extend(labels.cpu().numpy())
                    all_predictions_train.extend(predicted.cpu().numpy())

                accuracy_train = correct_train / total_train
                average_loss_train = total_loss_train / len(train_loader)
                precision_train = precision_score(all_labels_train, all_predictions_train, average='weighted')
                recall_train = recall_score(all_labels_train, all_predictions_train, average='weighted')


                results.append({
                    'Transformation_training': nombre_directorio,
                    'Accuracy_Train': accuracy_train,
                    'Loss_Train': average_loss_train,
                    'Precision_Train': precision_train,
                    'Recall_Train': recall_train,
                    'Transformation_eval': transformacion

                })

                # Guardar el progreso después de cada iteración
                df_progress = pd.DataFrame(results)
                df_progress = pd.concat([df_progress1,df_progress])
                df_progress.to_csv(progress_file, index=False)

# Guardar resultados finales en un CSV separado
df_final = pd.DataFrame(results)
df_final.to_csv(os.path.join(main_directory, 'performance_metrics_final_training.csv'), index=False)

print("Resultados finales guardados en performance_metrics_final_training.csv")


# %%
#%% correcion

df1 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_final_training.csv')
print(df1.shape)
df2 = pd.read_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_progress_training.csv')
print(df2.shape)
df = pd.concat([df1,df2]).reset_index(drop=True)
print(df.shape)
df.to_csv('/home/tbadaracco/models/efficientnet_b0/weights_none/performance_metrics_final_training.csv',index = False)

# %%
