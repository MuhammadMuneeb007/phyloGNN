 
#!/usr/bin/env bash
set -e

echo "========================================"
echo "STEP 3: INSTALL PHYLOCNN ENVIRONMENT"
echo "========================================"

# 1. Create the Python environment
echo "[1/6] Creating Conda environment"
mamba create -n phylocnn python=3.8.20 pip -c conda-forge -y

# 2. Activate environment
echo "[2/6] Activating phylocnn"
eval "$(conda shell.bash hook)"
conda activate phylocnn

# 3. Read dependencies from the ORIGINAL repository
echo "[3/6] Reading phyloCNN/environment.yml"
cd phyloCNN

sed -n '/^  - pip:/,$s/^      - //p' environment.yml \
  | grep -vE '^(tensorflow-macos|appnope)==' \
  > ../Step3-PhyloCNN-Linux.txt

cd ..

echo "Packages requested by original PhyloCNN:"
cat Step3-PhyloCNN-Linux.txt

# 4. Install all declared Python dependencies
echo "[4/6] Installing original PhyloCNN packages"
python -m pip install -r Step3-PhyloCNN-Linux.txt

# 5. Check dependencies
echo "[5/6] Checking installed packages"
python -m pip check

python -c "import tensorflow as tf; import keras, numpy, pandas, scipy, sklearn, ete3; print('TensorFlow:',tf.__version__); print('Keras:',keras.__version__); print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__); print('SciPy:',scipy.__version__); print('Scikit-learn:',sklearn.__version__); print('ETE3: OK')"

# 6. Install separate R environment for BiSSE
echo "[6/6] Installing R packages for BiSSE"
mamba create -n phylo-r -c conda-forge \
  r-base r-castor r-treesim -y

conda run -n phylo-r Rscript -e \
  'library(castor); library(TreeSim); cat("BiSSE R packages: OK\n")'

echo "========================================"
echo "STEP 3 COMPLETE"
echo "========================================"
echo "Python environment: phylocnn"
echo "R environment: phylo-r"
echo "To activate Python: conda activate phylocnn"
BASH
 