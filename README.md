# Cognitive_Coordinator
Cognitive Coordinator source code within the SAFE-6G project

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Train](#train)
- [Callibrate](#callibrate)

## Installation

```bash
git clone https://github.com/FRONT-research-group/Cognitive_Coordinator.git
```

## Usage
- Python 3.10.12
- SWI-Prolog version 8.4.2

```bash
cd Cognitive_Coordinator
python3 -m venv cc_env
source cc_env/bin/activate
pip install -r requirements.txt
```

## Train
To train the model that calculates the nLoTW run:

```bash
python3 main.py
```

## Callibrate
To test the calibrate script run:

```bash
python3 callibrate.py