# ForDead_package

## Installation
### Conda install (recommended)
Depuis un terminal, créer un nouvel environnement avec la commande suivante :

```bash
conda create --name ForDeadEnv python=3.7
```
Activer le nouvel environnement :
```bash
conda activate ForDeadEnv
```
Installer les packages numpy, geopandas, rasterio et scipy avec les commandes suivantes:
```bash
conda install numpy
conda install geopandas
conda install rasterio
conda install xarray
conda install scipy
```

Install rioxarray : https://corteva.github.io/rioxarray/stable/installation.html

Depuis l'invite de commande, placer vous dans le répertoire de votre choix et lancez les commandes suivantes :
```bash
git clone https://gitlab.com/jbferet/fordead_package.git
cd fordead_package
pip install .
```

## Utilisation
La détection du déperissement se fait en trois étapes.
- Le calcul des indices de végétation et des masques
- L'apprentissage par modélisation de l'indice de végétation pixel par pixel à partir des premières dates
- La détection du déperissement par comparaison entre l'indice de végétation prédit par le modèle et l'indice de végétation réel.

### Etape 3 : Main_DetectionForDead
L'étape de détection prend en entrée un dossier avec l'arborescence suivante:
* VegetationIndex 
    * VegetationIndex_YYYY_MM_JJ.tif
    * ...
* Mask
    * Mask_YYYY_MM_JJ.tif
    * ...
* DataModel
    * stackP
    * rasterSigma
* DataAnomalies

Un tel dossier est disponible dans le package ForDead_data.
La chaîne de traitement s'arrête pour le moment au calcul des anomalies.
