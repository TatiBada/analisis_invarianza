
from pylab import *
import torchvision
import torch
import torchvision.transforms.functional as TF

def rotation_transforms():

    rotation_angles = [30, 60, 90, 120, 150, 180, 210, 240]

    rotation_transforms = [lambda x, angle=angle: TF.rotate(x, angle) for angle in rotation_angles]

    return rotation_transforms


def translation_transforms():

    translation_values = [(10, 10), (20, 20), (-10, -10), (-20, -20), (5, 15), (-15, -5), (0, 0), (30, 30)]

    translation_transforms = [lambda x, dx=dx/2, dy=dy/2: TF.affine(x, angle=0, translate=(dx, dy), scale=1, shear=0) for dx, dy in translation_values]

    return translation_transforms


def scale_transforms():

    scale_factors = [0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.5]

    scale_transforms = [lambda x, scale=scale: TF.affine(x, angle=0, translate=(0, 0), scale=scale, shear=0) for scale in scale_factors]

    return scale_transforms

 
def perspective_transforms():

    perspective_params = [
        (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[5, 5], [5, 58], [58, 5], [63, 58]]  # Puntos finales
        ),
            (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[7, 7], [7, 56], [56, 7], [63, 56]]  # Puntos finales
        ),
        (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[10, 10], [10, 53], [53, 10], [63, 53]]  # Puntos finales
        ),
            (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[13, 13], [13, 50], [50, 13], [63, 50]]  # Puntos finales
        ),
        (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[15, 15], [15, 48], [48, 15], [63, 48]]  # Puntos finales
        ),
            (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[17, 17], [17, 46], [46, 17], [63, 46]]  # Puntos finales
        ),
        (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[20, 20], [20, 43], [43, 20], [63, 43]]  # Puntos finales
        ),
        (
            [[0, 0], [0, 63], [63, 0], [63, 63]],  # Puntos de inicio
            [[25, 25], [25, 38], [38, 25], [63, 38]]  # Puntos finales
        )
    ]

    perspective_transforms = [lambda x, startpoints=startpoints, endpoints=endpoints: TF.perspective(x, startpoints, endpoints) for (startpoints, endpoints) in perspective_params]

    return perspective_transforms

brightness_parameters = [0.25, 0.5, 0.75,1.25, 1.5, 2.0, 2.25, 2.75]

def brightness_transforms(brightness_factor:float):
    return lambda x: x + brightness_factor

#brightness_transforms = [brightness_transform(factor) for factor in brillo_list]


contrast_list = [ 0.3, 0.5, 0.7, 0.9,1,2.5,4,6]

def contrast_transforms(alpha):
    return lambda x: torchvision.transforms.functional.adjust_contrast(x, alpha) 

#contrast_transformations = [create_contrast_transform(alpha) for alpha in contrast_list]

def grayscale_transform(image):
    return TF.rgb_to_grayscale(image)


def interpolate_color_and_gray(color_image, gray_image, alpha):
    return alpha * color_image + (1 - alpha) * gray_image


grey_list = [0.05,0.1, 0.3, 0.5, 0.7, 0.9,1,1.5]

# Función para crear transformaciones de escala de grises con alpha variable
def grayscale_transforms(alpha):
    return lambda x: interpolate_color_and_gray(x, grayscale_transform(x), alpha)

#grayscale_transformations = [create_grayscale_transform(alpha) for alpha in grey_list]


def solarize_transforms(threshold: float):
    return lambda x: torch.where(x < threshold, x, 255 - x)

# Lista de umbrales para la solarización
solarization_thresholds = [5,3,1,0.7,0.5,0.2,0.05,0.001]

# Crear las transformaciones usando solarize_transform
#solarize_transformations = [solarize_transform(threshold) for threshold in solarization_thresholds]

#%%
#lista para posterizacion
posterize_list = [0.05,0.15,0.3, 0.5, 0.7, 0.9,1.5,2]

def posterize_transform(image, bits=4):
    shifts = 8 - bits
    image = (image * 255).to(torch.uint8)  # Convertir a tipo uint8 antes de los desplazamientos
    return ((image >> shifts) << shifts).to(torch.float32) / 255  # Convertir de vuelta a float32 después



# Función de interpolación para posterización
def interpolate_posterization(color_image, transformed_image, alpha):
    return alpha * color_image + (1 - alpha) * transformed_image


def posterize_transforms(alpha):
    return lambda x: interpolate_posterization(x, posterize_transform(x), alpha)

# Generar la lista de transformaciones de escala de grises con alpha variable
#posterize_transformations = [create_posterize_transform(alpha) for alpha in posterize_list]

#%%

#lista para inversion de colores
invertion_list = [ -0.4,-0.3, -0.1, 0,0.05 ,0.1,0.3,0.5]
#invertion_transformations = [ torchvision.transforms.functional.adjust_hue(dataset_nolabels[i], alpha) for i in range(0, n, step) for alpha in invertion_list]

def invertion_transforms(alpha):
    return lambda x: torchvision.transforms.functional.adjust_hue(x, alpha) 

#invertion_transformations = [create_invertion_transform(alpha) for alpha in invertion_list]


