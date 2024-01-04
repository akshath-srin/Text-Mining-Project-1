# Chinese Named Entity Recognition

## Dataset
This project explores various models (HMM, CRF, Bi-LSTM, Bi-LSTM+CRF) for Chinese Named Entity Recognition. The dataset, from the "Chinese NER using Lattice LSTM" ACL 2018 paper, comprises resume data. Data format: each line contains a character and its corresponding label using the BIOES tagging system. Sentences are separated by blank lines.

Example:

```
美	B-LOC
国	E-LOC
的	O
华	B-PER
莱	I-PER
士	E-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

The dataset is located in the `ResumeNER` folder.

## Results
Prediction results of these four models :

| Model       | Recall  | Precision | F1 Score |
|-------------|---------|-----------|----------|
| HMM         | 91.22%  | 91.49%    | 91.30%   |
| CRF         | 95.43%  | 95.43%    | 95.42%   |
| BiLSTM      | 95.34%  | 95.38%    | 95.30%   |
| BiLSTM+CRF  | 95.62%  | 95.66%    | 95.62%   |


## Quick Start

Cloning the repository

```
git clone https://github.com/akshath-srin/Text-Mining-Project-1.git
```

Create and activate conda environment

```
conda create --name chinese-ner python=3.8

conda activate chinese-ner
```

Install dependencies:

```
pip install -r requirement.txt
```

Install PyTorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Train and evaluate the model:

```
python main.py
```

Model and training parameters can be adjusted in ./models/config.py.

To load and test a trained model:

```shell
python test.py
```
