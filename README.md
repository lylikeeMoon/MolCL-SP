# MolCL-SP: Reproduction Instructions

This repository provides the source code for reproducing the MolCL-SP model.

## 🚀 Clone This Project

```bash
git clone https://github.com/likeeMoon/MolCL-SP.git
cd MolCL-SP
```

## 📁 Dataset and Pre-trained Models

Our pre-trained MolCL-SP model, RoBERTa model, and datasets can all be downloaded via Baidu Netdisk:

**Download link:** [https://pan.baidu.com/s/1H78\_tIKdDU-FuE2Yr7vYTg](https://pan.baidu.com/s/1H78_tIKdDU-FuE2Yr7vYTg)
**Extraction code:** `p46y`

Please download and extract the files before training or fine-tuning.

Further data processing can be done by running:

```bash
python pcqm4m.py
```

## 🧪 Pre-training

To pre-train the model from scratch, run:

```bash
python pretrain.py
```

Alternatively, you can directly load our pre-trained model provided in the downloaded files.

## 🎯 Fine-tuning

To fine-tune the model on a specific downstream task, specify the task name in `finetune.py`, then run:

```bash
python finetune.py
```

## 📌 Notes

* Ensure all dependencies are installed (via `environment.yml` or `requirements.txt`).
* GPU is recommended for faster training and fine-tuning.

---

If you find this project helpful, please consider starring the repository 🌟
