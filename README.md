## PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree
[Finding of NAACL 2025](https://aclanthology.org/2025.findings-naacl.464/)

### Installation
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```

### Train
```shell
python build_graph.py
python gcn.py
```

### Detect
```shell
python detect --text "This is an example."
```

### Citation
```bibtex
@inproceedings{li2025prdetect,
  title={PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree},
  author={Li, Xiang and Yin, Zhiyi and Tan, Hexiang and Jing, Shaoling and Su, Du and Cheng, Yi and Shen, Huawei and Sun, Fei},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={8290--8301},
  year={2025}
}
```
