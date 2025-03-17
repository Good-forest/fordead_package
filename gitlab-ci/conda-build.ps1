$condaurl = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
# Path where miniconda installer will be downloaded
$condaexe = "${env:userprofile}\Miniforge3-Windows-x86_64.exe"
# Path where miniconda will be installed
$condadir = "${env:userprofile}\miniforge3"

if ( -not (Test-Path -Path $condadir -PathType Container) ){
    echo "Installing Conda in: $condadir"
    # download miniconda installer
    curl.exe -L -C - $condaurl -o $condaexe
    # install miniconda
    Start-Process $condaexe -Args "/InstallationType=JustMe /RegisterPython=0 /S /D=$condadir" -wait
    # add condabin to path to have `conda` command available
    $env:PATH = "$condadir\condabin;" + $env:PATH
    # initialize conda: it creates the powershell profile script
    conda init powershell
}
# load the profile for current session: it activates (base) environment
invoke-expression -Command "$env:userprofile\Documents\WindowsPowerShell\profile.ps1"
