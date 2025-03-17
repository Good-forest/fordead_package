# Downloading Sentinel-2 data

Though this package is meant to be used with Sentinel-2 data from any provider, it has been mainly used and tested using MAJA data and it provides tools to easily download and preprocess the required data.

This tool can be used either from a script or from the command line, and automatically downloads all Sentinel-2 data from GEODES plateform between two dates under a cloudiness threshold. Then this data is unzipped, keeping only chosen bands from Flat REflectance data, and zip files are emptied as a way to save storage space.

It can occur that the same Sentinel-2 scene (date & tile) is splitted in two archives when the scene is covered by two granules.
In that case, the two pieces of raster are merged mosaicing with the average (ignoring no-data) and written in one of the duplicates,
the others being removed. Still, the empty zip files of the merged scenes are kept in order to keep track of that operation.

## Authentication
In order to have download access to GEODES plateform, you will
need to create an account at https://geodes.cnes.fr/ and create
an API key on that account.

The API key should then be specified in the section `geodes`
of the EODAG config file "$HOME/.config/eodag/eodag.yaml".
It should look like this:
```yaml
geodes:
    priority: # Lower value means lower priority (Default: 0)
    search: # Search parameters configuration
    auth:
        credentials:
            apikey: "write your api key here"
    download:
        extract:
        output_dir:
```

If $HOME/.config/eodag/eodag.yaml does not exists,
run the following in a python session, it should create the file:
```python
from path import Path
from eodag import EODataAccessGateway

# creates the default config file at first call
EODataAccessGateway()

# prints the path to the default config file, makes sure it exists
config_file = Path("~/.config/eodag/eodag.yml").expand()
print(config_file)
assert config_file.exists()
```

If the geodes section does not exist in the config file,
it must be that the config file template is not up to date
with EODAG v3+. In order to fix it, rename the file
"$HOME/.config/eodag/eodag.yaml" to "$HOME/.config/eodag/eodag.yaml.old"
and create a new with the above code.

If there are special characters in your API key,
make sure to add double quotes around it.

## Examples
### From a script

```bash
from fordead.cli.cli_theia_preprocess import theia_preprocess
theia_preprocess(
    zipped_directory = <zipped_directory>,
    unzipped_directory = <unzipped_directory>,
    tiles = ["T31UFQ","T31UFP"],
    level = "LEVEL2A",
    start_date = "2015-01-01",
    end_date = "2025-01-01",
    lim_perc_cloud = 50,
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"])
```

### From the command line

```bash
fordead theia_preprocess -i <zipped_directory> -o <unzipped_directory> -t T31UFQ -t T31UFP --level LEVEL2A --start_date 2015-01-01 --end_date 2025-01-01 -n 50 -b B2 -b B3 -b B4 -b B5 -b B6 -b B7 -b B8 -b B8A -b B11 -b B12 -b CLMR2 --dry-run
```

See detailed documentation on the [site](https://fordead.gitlab.io/fordead_package/docs/cli/#fordead-theia_preprocess)