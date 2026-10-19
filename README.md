
## 环境配置

以下命令用于 Linux、Python 3.10。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 数据准备

使用自己的单通道 MRI 数据，背景标签为 `0`，前景标签为 `1`。文件按以下结构存放，数据目录名称以 `Dataset` 开头：

```text
Dataset/
├── dataset.json
├── imagesTr/CASE_0000.nii.gz
└── labelsTr/CASE.nii.gz
```

参考 `configs/dataset.example.json` 编写数据目录中的 `dataset.json`，将 `numTraining` 改为实际病例数，至少需要 5 个独立病例。工作目录放在源码和原始数据目录之外。

```bash
RAW=/path/to/Dataset
WORK=/path/to/Dataset_adaptive_run
GPU=0

python scripts/prepare.py --workspace "$WORK" --raw "$RAW" 
python tests/verify.py --workspace "$WORK" --gpu "$GPU" --output "$WORK/verification.json"
```

预处理会自动生成当前数据的指纹、3D 配置和五折划分，全部保存在 `$WORK`。划分和训练默认种子为 `20261017`，每折训练种子加 `10000 × 折号`，可用 `--seed` 指定。若同一患者有多次扫描，预处理时加 `--groups /path/to/groups.json`，文件格式为 `{"CASE_A": "patient_1", "CASE_B": "patient_1"}`，需覆盖全部病例。

## 训练

```bash
python scripts/train.py --workspace "$WORK" --raw "$RAW" \
  --variant adaptive --fold 0 --gpu "$GPU" \
  --epochs 700 
```

`--fold` 可选 `0`～`4`。按顺序运行全部五折：

```bash
bash scripts/run_five_folds.sh "$WORK" "$RAW" "$GPU"
```

继续中断的训练时使用相同参数并加 `--resume`。训练配置、划分、随机种子、日志及评估结果保存在工作目录，模型和默认损失函数不变。


## 评估与预测

每折训练结束后，程序使用 `checkpoint_final.pth` 对完整验证图像计算 Dice、Precision、Recall，并在 `$WORK/release/` 导出仅含权重及必要配置的预测包。重新评估使用训练目录中已有的验证预测和标签；下载的权重包只用于预测。

```bash

python scripts/evaluate.py --model-folder /path/to/model --labels /path/to/labels --fold 0


python scripts/predict.py --model-folder /path/to/model \
  --input /path/to/images --output /path/to/predictions --folds 4 --gpu "$GPU"
```

请将示例路径替换为实际路径。评估命令要求模型目录中已有 `fold_0/validation/` 下的预测结果。
新图像按 `CASE_0000.nii.gz` 命名。预测时将 `--model-folder` 指向权重包解压后的目录；公开的第 4 折权重用 `--folds 4`，自己的其他折按实际折号填写。权重下载位置见 `WEIGHTS.md`。

