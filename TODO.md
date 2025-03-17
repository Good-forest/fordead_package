# next version
- [ ] theia_preprocess.maja_download : look at duplicated dates only on unzip.
# recent
- [x] split current_state in {current_state and static or timeless} including
  model coeffs
- remove dependency to time in extract_raster_values
- [x] fix issue with id_point
- [x] chunks/chunksize in step5
- warning for max nb points in extract_raster_values
- add by_chunk in command lines (extract_reflectance, step5)
- place all test and example data (fordead_data, outputs) in a data directory to
  simplify `.gitignore` and cleaning
- [x] issue in tests calling `points.to_file(Path(...))` only on Windows: 
  rasing `TypeError: Expected unicode, got Path`
  seems to be due to a direct call to cython function expecting ctype str
- [x] remove constraint on python version as a workaround was found to issue https://github.com/microsoft/vscode-python-debugger/issues/498

# Tester la construction de la collection sur un gros jeu de données (1000 images)

# ajouter central_wavelength et bandwidth au assets
i.e. simplifier la configuration en partant d'un json ou d'un dictionnaire comme
dans gdalcubes ou eoreader.

# wavelength ou common_name pour le calcul des indices
Il est possible d'utiliser la wavelength (nearest de la wavelength souhaitée) ou
les common_name (green, red, rededge, nir, ...) pour le calcul des indices.
Ca permettrait d'être indépendant du capteur.

# Fonctions et classes

Dans la description ci-dessous, les input ont les dimensions nécessaires, voir s'il est possible de garder les mêmes fonctions mais avec des dimensions extra, i.e. en propageant les dimensions supplémentaires au résultat: apply along ou boucle for.

```python
compute_mask(xr[x, y, band], formula="B2>700, use_CLM=True) -> xr[x,y,band="FORDEAD_MASK"]

compute_soil(xr[x, y, band], formula="B5>700") -> xr[x,y,band="FORDEAD_SOIL"]

compute_vi(xr[x, y, band], vi="CRSWIR", formula=None) -> xr[x,y,band="FORDEAD_VI_{indice}"]
    si indice is None:
        Error("vi name must be specified")
    
    si indice in indice_db:
        si formula not None:
            Error("Indice name already in indice_db")
        sinon
            formula = indice_db[indice]
    
mask_vi(xr[x,y,band], vi="CRSWIR", mask_bands=["MASK"]) -> xr[x,y]

train_model(xr[x, y, band, t], vi="CRSWIR", ...) -> xr[x,y, band=[alpha_1, alpha_2, ..., enddate]] raster of parameters of the model and enddate
    mask_vi(xr[x,y,band,t], vi="CRSWIR", mask_bands=["MASK", "SOIL"])
    ...

predict(xr[x, y, band]) -> xr[x,y] anomaly raster
    mask_vi(xr[x,y,band,t], vi="CRSWIR", mask_bands=["MASK"])

add_asset(coll, xr[x,y,band,time], output_dir, band_parser):
    for t in xr.time:
        item = coll.filter(datetime=t)[0]
        for b in xr.band:
            band_file = write_tif(xr.sel(band=b), output_dir / item.id+f'_{b}.tif')
            asset = band_parser(band_file)
            item.add_asset(asset)


# en cours de construction
Class FordeadApp(config, coll=None):

    def __init__(self)
        if coll is not None:
            self.coll=coll
            config.pop("input_dir", None)
        self.config = config

    @class_method
    def from_file(cls, config_file):

        config = load(config_file)
        coll = load_coll
```

