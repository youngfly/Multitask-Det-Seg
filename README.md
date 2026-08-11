# README — 检测 / 分割 多任务训练·推理·部署框架

> 基于 ultralytics/yolov5（tag v5.0）改造的**单模型多任务框架**：一个 YOLOv5 主干**并联一个轻量分割头**，一次前向同时输出**目标检测框 + 语义分割图**。
> 本文档按**当前工作区真实文件关系**撰写，训练/推理示例统一以 **`dataset`**（原 `dataset2`）数据集为准。

---

## 0. 项目定位

一个面向**检测 + 分割联合任务**的框架，覆盖从**数据准备 → 训练 → 推理 → RTSP 部署**的完整链路：

| 阶段     | 入口                          | 说明                                                       |
| -------- | ----------------------------- | ---------------------------------------------------------- |
| 数据准备 | `tools/*.py`                | 标注 json → YOLO txt、分割掩码改名、可视化                |
| 训练     | `train.py`                  | 双 DataLoader 多任务训练，输出`runs/train/exp*/weights/` |
| 推理     | `detect.py` / `test.py`   | 图片/视频/RTSP，输出`[检测框, 分割图上色]`               |
| 部署     | `main.py` / `main_new.py` | RTSP 拉流 → 推理 → 回推，用于边缘盒子                    |

**任务语义（`dataset`）**

- **检测 `nc=2`**：`戴口罩` / `不带口罩`（`bdd_det_seg.yaml` 中 `names=['0','1']` 为占位名，真实语义见上）。
- **分割 `n_segcls=5`**：`背景 + 桌面 + 人体戴口罩 + 地面 + 人体不带口罩`（详见 §4）。

---

## 1. 核心特性

- **单模型双输出**：一次前向同时完成检测与分割，比串行使两个独立模型更省显存。
- **4 种可切换分割头**（`models/yolo.py`）：`SegMaskBase` / `SegMaskBiSe` / `SegMaskLab` / `SegMaskPSP`，默认启用 **`SegMaskPSP`**（精度与速度平衡最好，详见 §3）。
- **多任务损失加权**：`detgain=0.6`、`seggain=0.35`（硬编码于 `train.py`）。
- **梯度累积**：名义 batch 固定 `nbs=64`，按实际总 batch 自动 `accumulate = floor(64 / total_batch_size)`。
- **混合精度训练**：`torch.cuda.amp.GradScaler`。
- **RTSP 端到端部署**：`main.py` 支持推流/拉流/回推推理。

---

## 2. 当前目录结构

```
.
├── detect.py              # 推理（图片/视频/RTSP/提交/测速）
├── test.py                # 验证：分割 mIoU + 检测 mAP（被 train.py import）
├── train.py               # 训练主循环（dataset 多任务）
├── main.py / main_new.py  # RTSP 推拉流部署推理
├── SegmentationDataset.py # 分割数据加载（BddSegmentation / get_bdd_loader / mask_generate）
├── hubconf.py / logger.py / Dockerfile
├── data/
│   ├── bdd_det_seg.yaml   # ★ dataset 训练配置（train/val/segtrain/segval → ./dataset）
│   ├── hyp.scratch.yaml / hyp.finetune.yaml
│   └── coco、cityscapes、custom 等其它配置
├── models/
│   ├── yolov5s_city_seg.yaml  # ★ 默认模型结构（nc=2, n_segcls=5, SegMaskPSP+Detect）
│   ├── yolov5m_city_seg.yaml / yolov5m_citybdd.yaml / yolov5s_custom_seg.yaml
│   ├── yolo.py             # Model 类：前向返回 [det_out, seg_out]
│   └── common.py / experimental.py / export.py
├── utils/
│   ├── datasets.py         # 检测 DataLoader（create_dataloader）
│   ├── loss.py             # ComputeLoss（检测）+ SegmentationLosses（分割）
│   └── general.py / plots.py / autoanchor.py / torch_utils.py / metrics.py ...
├── dataset/               # ★ 训练数据集（核心，原 dataset2）
│   ├── detdata/images/{train,val}/   # 检测图像（各 396 张 jpg）
│   ├── detdata/labels/{train,val}/   # YOLO 检测标签（各 396 个 txt）
│   ├── segimages/{train,val}/         # 分割图像（各 396 张 jpg）
│   ├── seglabels/{train,val}/          # 分割彩色 mask（train 396 / val 397 个 png）
│   ├── json_det/                        # 标注工具导出的原始检测 json（396 个）
│   ├── video/                           # 演示视频（1.mp4，原根目录 video/）
│   └── dataset_extra/                  # 另一批次原始标注导出（原 dataset2/dataset3；json_det/json_seg/png/png_seg，未接线）
├── tools/                 # 数据准备 + 可视化脚本
│   ├── peojson2yolo.py    # json 检测标注 → YOLO .txt（输出 ./dataset/yolodet/）
│   ├── fix_name.py        # 分割标签改名，使其与 segimages/*.jpg 对应
│   ├── sz_gen_gt.py       # 预转换 mask（loader 实际在线转换，产物未被使用）
│   ├── vis.py / vis_mask.py  # 可视化（读 dataset/dataset_extra/）
│   └── camera_v5.py       # 摄像头推流辅助
├── archive/               # 已归档、dataset 流程不使用的变体
│   ├── train_citysbdd.py  # Cityscapes+BDD 混合训练（n_segcls=19）
│   ├── train_custom.py    # 自定义单类分割训练（n_segcls=19）
│   └── test_custom.py     # test.py 的 Cityscapes/custom 验证变体
├── weights/               # 仅 download_weights.sh（yolov5s/m/l/x.pt 由脚本下载，不随仓库）
├── runs/                  # 训练产物（exp*/weights/{last,best}.pt）
├── .github/               # CI/社区模板：workflows(ci-testing/codeql/greetings/rebase/stale)、ISSUE 模板、dependabot
├── .gitignore / .gitattributes / .dockerignore / LICENSE / requirements.txt
└── README.md              # 本文档
```

---

## 3. 模型架构

### 3.1 前向契约

`models/yolo.py` 的 `Model.forward_once` 在最后**返回 `[检测输出, 分割输出]`**：

```python
out = model(img)          # 单模型一次前向，out 是一个列表
det = out[0][0]           # 检测：经 NMS 后的框  [B, N, 5+nc]
seg = out[1]              # 分割：上采样到原分辨率的 mask  [B, n_segcls, H, W]
```

- 模型构建时 `self.save.append(24)` 硬编码记录分割层（Detect 是第 25 层，必须作为 YAML 的**最后一层**；若移动分割头位置，须同步修改此行）。
- `parse_model` 对 `SegMask*` 头单独处理通道缩放（`models/yolo.py` 约 407 行）。

### 3.2 四种分割头对比（默认 SegMaskPSP）

| 头              | 结构来源            | 特点                                          | 推荐度                 |
| --------------- | ------------------- | --------------------------------------------- | ---------------------- |
| `SegMaskBase` | C3 + SPP            | 结构最简单，单输入                            | 一般                   |
| `SegMaskBiSe` | BiSeNetV1 风格      | 含 aux loss，接口未统一，需手动改`train.py` | 不推荐                 |
| `SegMaskLab`  | DeepLabV3+ 风格     | FFM 注意力融合浅深层                          | 推荐                   |
| `SegMaskPSP`  | PSPNet 风格（默认） | RFB2 + PyramidPooling + FFM，精度/速度均衡    | **推荐（默认）** |

切换方式：编辑 `models/yolov5s_city_seg.yaml` 的 head 段，取消对应行的注释即可（见该文件 48–51 行）。

### 3.3 多任务损失

硬编码于 `train.py`：

```python
detgain, seggain = 0.6, 0.35                 # 检测 / 分割 损失权重
compute_loss    = ComputeLoss(model)          # 检测：CIoU + BCE
compute_seg_loss = SegmentationLosses(aux=False, ignore_index=-1)  # 分割：CrossEntropy
...
loss *= detgain                               # 检测分支
segloss *= seggain                            # 分割分支（CE 均值需 * batch_size 配合梯度累积）
total = loss + segloss
```

---

## 4. 数据集 `dataset`

### 4.1 目录与格式

`dataset/` 同时提供**检测**与**分割**两份数据，分别由两个 DataLoader 读取：

| 子目录                          | 内容                                                       | 被谁读取                                               |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| `detdata/images/{train,val}/` | 检测原图 jpg                                               | `utils.datasets.create_dataloader`（`train_path`） |
| `detdata/labels/{train,val}/` | YOLO 检测标签 txt（每行`cls cx cy wh`）                  | 同上                                                   |
| `segimages/{train,val}/`      | 分割原图 jpg（与检测原图是**同一批图**，仅布局不同） | `get_bdd_pairs`                                      |
| `seglabels/{train,val}/`      | 分割彩色 mask png（RGB 配色，运行时转 5 类）               | `BddSegmentation`                                    |
| `json_det/`                   | 标注工具导出的原始检测 json                                | 数据准备阶段（一次性）                                 |

### 4.2 类别约定

**检测 `nc=2`**（`names=['0','1']` 为占位名）：

| 检测类 id | 语义     |
| --------- | -------- |
| 0         | 戴口罩   |
| 1         | 不带口罩 |

**分割 `n_segcls=5`**（`BddSegmentation.mask_generate` 按 **R+G 通道和** 把彩色 mask 映射为类别 id）：

| 分割类 id | 语义         | 存储 mask RGB     | R+G 和 | 可视化颜色（peotable_COLORMAP） |
| --------- | ------------ | ----------------- | ------ | ------------------------------- |
| 0         | 背景         | —                | 0      | `[70,70,70]`                  |
| 1         | 桌面         | `[0,85,255]`    | 85     | `[255,255,0]`                 |
| 2         | 人体戴口罩   | `[0,170,0]`     | 170    | `[0,85,255]`                  |
| 3         | 地面         | `[127,127,127]` | 254    | `[0,170,0]`                   |
| 4         | 人体不带口罩 | `[255,85,0]`    | 340    | `[204,43,41]`                 |

- 分割忽略类 `ignore_index=-1`；mask 中 pad 用 255（在 `mask_generate` / test 中置 -1）。
- **配置一致性**：`bdd_det_seg.yaml` 的 `nc:2` ↔ `yolov5s_city_seg.yaml` 的 `nc:2, n_segcls:5` ↔ `train.py` 中 `n_segcls=5` **三方一致**（改数据/模型时务必保持同步）。

### 4.3 数据准备（`tools/`，一次性）

| 脚本                      | 输入                                    | 输出                         | 说明                                                                                                                                |
| ------------------------- | --------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `tools/peojson2yolo.py` | `dataset/json_det/*.json`             | `dataset/yolodet/*.txt`    | json 检测框 → YOLO txt。**注意输出到 `yolodet/`，需手动把 `.txt` 归位到 `detdata/labels/train\|val`**                   |
| `tools/fix_name.py`     | `dataset/seglabels/val/*_seg.png`     | `*.png`                    | 让分割标签文件名与`segimages/*.jpg` 对应（`get_bdd_pairs` 用 `replace('segimages','seglabels').replace('.jpg','.png')` 匹配） |
| `tools/sz_gen_gt.py`    | `dataset/seglabels/*.png`             | `dataset/seglabels2/*.png` | 预转换 mask；但 loader 在线转换，产物实际未被使用                                                                                   |
| `tools/vis.py`          | 原图 +`json_det`                      | 检测框可视化                 | 人工检查用                                                                                                                          |
| `tools/vis_mask.py`     | 原图 +`dataset/dataset_extra/png_seg` | 分割叠加可视化               | 人工检查用（注：脚本以`../dataset/dataset_extra/png_seg/` 为输入；文件名会落为`*.jpg.jpg`，含后缀 bug）                         |

### 4.4 数据集下载（ModelScope）

本数据集已发布到 ModelScope，供他人直接获取（约 1.1 GB / 7500+ 文件）：

- 仓库地址：<https://www.modelscope.cn/datasets/learnai2/multiyolo-det-seg>
- 数据集说明（标注格式、类别约定、掩码解码等）：见仓库内 `README.md` 或本仓库 `dataset/README.md`

**方式一：git 克隆（需 git-lfs）**

```bash
git clone https://www.modelscope.cn/datasets/learnai2/multiyolo-det-seg.git
# 克隆后得到 multiyolo-det-seg/ 目录，将其内容放到本项目的 dataset/ 目录即可
#   （即让 dataset/detdata、dataset/segimages、dataset/seglabels 等就位）
```

**方式二：ModelScope SDK 下载（推荐，可断点、可指定缓存路径）**

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

dataset_dir = snapshot_download(
    'learnai2/multiyolo-det-seg',
    repo_type='dataset',          # 数据集仓库，必须指定
)
print(dataset_dir)                # 本地缓存路径，如 ~/.cache/modelscope/...
# 将 dataset_dir 内的内容复制/软链到本项目的 dataset/ 目录
```

**方式三：网页手动下载**

打开上述仓库地址 → 点击「下载」按钮，按页面指引下载 zip 并解压到本项目的 `dataset/` 目录。

**方式四：命令行 CLI 下载（modelscope 命令行，推荐）**

需先安装 `modelscope`（与 SDK 同包）：`pip install modelscope`。安装后即可用 `modelscope download` 命令直接拉取：

```bash
# 下载整个数据集到 ./dataset（--local_dir 指定落地目录）
modelscope download --dataset learnai2/multiyolo-det-seg --local_dir ./dataset

# 也可只下载单个文件（如数据集 README）先确认内容，再拉全量
modelscope download --dataset learnai2/multiyolo-det-seg README.md --local_dir ./dataset
```

> 提示：`modelscope download --local_dir ./dataset` 会把数据**平铺**到指定的 `./dataset` 目录下（不额外套 `learnai2/multiyolo-det-seg/` 子目录），正好与 `data/bdd_det_seg.yaml` 约定的 `./dataset/detdata`、`./dataset/segimages`、`./dataset/seglabels` 等路径对齐，无需再手动挪动。命令行方式相比 git 克隆更轻量、支持断点续传。

> 下载完成后，请确保项目目录结构满足 `data/bdd_det_seg.yaml` 的约定（`./dataset/detdata/images/...`、`./dataset/seglabels/...` 等），否则训练会找不到数据。

---

## 5. 环境配置

```bash
pip install -r requirements.txt
pip uninstall wandb        # 必须！仓库不支持多卡 / wandb，不卸载会 import 报错
```

依赖见 `requirements.txt`（torch≥1.7.0、opencv-python、matplotlib、tensorboard 等）。Docker 镜像基于 `nvcr.io/nvidia/pytorch:21.03-py3`（`Dockerfile`）。

---

## 6. 配置与模型关系

```
                 data/bdd_det_seg.yaml
                 (train→detdata/images, segtrain/segval→dataset, nc=2)
                          │
                          ▼
   models/yolov5s_city_seg.yaml  ──►  Model(cfg, nc=2, n_segcls=5)
   (SegMaskPSP + Detect 必须最后)          │
                                           │ 实例化
                                           ▼
                 ┌──────────────── train.py ────────────────┐
                 │                                          │
   检测分支:  create_dataloader(detdata/images)          分割分支:
    imgs → model(imgs)                                   get_bdd_loader(dataset)
    loss_det = ComputeLoss(pred[0]) * 0.6                 → BddSegmentation
                 │                                        → get_bdd_pairs(segimages+seglabels)
                 │                                        → mask_generate(RGB→5类)
                 │                                        segimgs → model(segimgs)
                 │                                        seg_loss = SegmentationLosses(pred[1]) * 0.35
                 │                                          │
                 │  总 loss = loss_det + seg_loss → backward/accumulate
                 │                                          │
                 │  验证: test.seg_validation(n_segcls=5) + test.test() mAP
                 └──────────────►  runs/train/exp*/weights/{last,best}.pt

   推理:  detect.py / test.py 加载 best.pt → out = model(img) → [det_out, seg_out]
   部署:  main.py (RTSP 推流推理)
```

---

## 7. 训练

> **前置**：预训练 backbone 权重**不在仓库内**（`weights/` 仅含 `download_weights.sh`，便于上传 GitHub）。训练前先下载：
>
> ```bash
> bash weights/download_weights.sh    # 下载 yolov5s/m/l/x.pt 到 weights/
> ```

```bash
pip uninstall wandb

CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data ./data/bdd_det_seg.yaml \
  --cfg  ./models/yolov5s_city_seg.yaml \
  --weights ./weights/yolov5s.pt \
  --batch-size 18 \
  --epochs 200 \
  --workers 0 \
  --label-smoothing 0.1 \
  --img-size 832 \
  --noautoanchor
```

- 输出：`runs/train/exp*/weights/{last.pt, best.pt}`。
- 训练集/验证集按 `detdata/images` 读检测、`dataset` 读分割，二者在 `train.py` 内独立建 loader。
- 验证指标：`test.seg_validation(n_segcls=5)`（pixAcc/mIoU）+ `test.test()`（检测 mAP），选优依据 `fitness2(mAP, mIoU)`。

---

## 8. 推理

### 8.1 图片

```bash
python3 detect.py \
  --weights ./runs/train/exp1/weights/best.pt \
  --source  ./dataset/detdata/images/val \
  --conf 0.25 \
  --img-size 512
```

### 8.2 视频

```bash
python3 detect.py \
  --weights ./runs/train/exp1/weights/best.pt \
  --source  ./dataset/video/1.mp4 \
  --conf 0.25 \
  --img-size 1024
```

`detect.py` 内部：`out = model(img, augment=...)` → `pred = out[0][0]`（检测）、`seg = out[1]`（分割），分割用 `label2image(seg, peotable_COLORMAP)` 上色叠加。支持 `--submit` 输出 Cityscapes 提交格式（`trainid2id`）。

### 8.3 验证 / 测试

```bash
python3 test.py      # 同时算分割 mIoU 与检测 mAP（使用对应 cfg/data/weights）
```

---

## 9. 部署（RTSP 拉推流）

三段式：**本地 `ffmpeg` 推流 → 盒子 `main.py` 拉流推理并回推 → 本地 `ffplay` 拉结果流**。

**① 推流**（把本地视频循环推到 RTSP 服务器）

```bat
.\ffmpeg.exe -stream_loop -1 -re -i ./dataset/video/1.mp4 ^
  -vcodec h264 -acodec aac -strict -2 -tune zerolatency ^
  -f rtsp -rtsp_transport tcp -g 25 rtsp://192.168.50.XXX:8554/live/test1
```

**② 盒子推理**（拉流 → 模型推理 → 回推，需改 `main.py` 三处：权重路径 / 接收地址 / 写出路径）

```bash
python main.py
```

**③ 本地拉结果流**

```bat
.\ffplay.exe rtsp://192.168.50.XXX:8533/live/detection -rtsp_transport tcp
```

> 上述 `192.168.50.xxx` 等为**部署示例占位 IP**，使用时按你的网络拓扑替换真实内网地址。

参考项目

[github.com/TomMao23/multiyolov5](https://github.com/TomMao23/multiyolov5)
