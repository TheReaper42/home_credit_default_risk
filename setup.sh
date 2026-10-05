set -e

python3 -m venv local_env

source local_env/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt