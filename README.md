# fordead : un package python pour la détection de changement en forêt à partir d'images SENTINEL-2

Le package `fordead` a été développé pour la détection de changements en forêt à partir de série temporelles SENTINEL-2, en particulier dans un contexte de crise sanitaire du scolyte sur les épicéas. A partir de fonctions simplifiant l'utilisation de données satellites SENTINEL-2, il permet la cartographie des déperissements liés aux scolytes depuis 2018. Cependant, excepté les masques utilisés qui sont certainement trop spécifiques à l'étude de peuplements résineux, le reste des étapes du processus peuvent également être utilisées pour d'autres contextes. 

## Installation
### Conda install (recommended)

Depuis l'invite de commande, placer vous dans le répertoire de votre choix et lancez les commandes suivantes :
```bash
git clone https://gitlab.com/raphael.dutrieux/fordead_package.git
cd fordead_package
conda env create --name fordead_env
conda activate fordead_env
pip install .
```

## Utilisation pour la détection de déperissement
La détection du déperissement se fait en cinq étapes.
- Le calcul des indices de végétation et des masques pour chaque date SENTINEL-2
- L'apprentissage par modélisation de l'indice de végétation pixel par pixel à partir des premières dates
- La détection du déperissement par comparaison entre l'indice de végétation prédit par le modèle et l'indice de végétation réel
- La création du masque forêt, qui définit les zones d'intérêt
- L'export de sorties permettant de visualiser les résultats au pas de temps souhaité

L'ensemble de la documentation ainsi que les guides utilisateurs de ces étapes sont disponibles sur le [site](https://raphael.dutrieux.gitlab.io/fordead_package/).
