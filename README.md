## Initiallization

```bash
sudo apt update
sudo apt install build-essential
sudo apt install python3-pip
sudo apt install python3-venv
sudo apt install git
```

## venv

```bash
python3 -m venv <your-env-name>
source <your-env-name>/bin/activate
```

## Install packages

- Install `torch` first

```python
pip3 install torch torchvision torchaudio
```

- Install packages

```bash
sh install_packages.sh
```

- Run
```bash
python  ./result.py
```
