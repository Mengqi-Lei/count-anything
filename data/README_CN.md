# CLOC 数据集发布说明

<p align="right">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

本文说明如何准备完整的 CLOC 数据集。我们提供 CLOC 的全部标注文件，以及一部分可以直接发布的增强图片；用户需要按照本文准备各个源数据集的原始图片，再运行统一的转换、重建和审计脚本。完成这些步骤后，即可得到能够被 CLOC 数据集标注完整索引的数据集工作区。

## CLOC 标注和增强图片包

请先从以下链接下载 CLOC 标注包和可直接发布的增强图片包：

- [Google Drive](https://drive.google.com/drive/folders/1jt_CLqfQRVV9Vzll2Aqlf1tQNbWbezSe?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1cCFxHLcU6o1EFGjtxZgseg?pwd=cloc)

下载 `cloc_annotations.zip` 和 `cloc_augmented.zip` 后，将两个压缩包放到 `data/` 根目录下：

```text
data/
  cloc_annotations.zip
  cloc_augmented.zip
```

然后在 `data/` 根目录执行解压命令：

```bash
cd /path/to/data
unzip -q cloc_annotations.zip
unzip -q cloc_augmented.zip
```

解压后应得到：

```text
data/
  annotations/
  augmented/
```

## 目录结构

```text
data/
  annotations/        # CLOC 数据集 JSON，image_path 使用本目录内的相对路径
  tools/              # 总脚本及其子目录；根目录只保留转换、重建、审计入口
  metadata/           # 脚本运行报告和审计结果
  images/             # 原始数据集压缩包和解压后的文件夹，按数据集名称组织
  augmented/          # CLOC 数据集标注引用的增强图片
```

其中 `annotations/` 的主要内容如下：

```text
annotations/
  *_split.json                 # 原始 train/val/test 划分，按图片记录组织
  *_expanded_by_class.json     # 将 split 标注按类别展开，使每条标注记录只包含一个目标类别
  class_group.json             # 类别 group 划分
  seen_unseen_split.json       # seen/unseen 类别划分
```

`augmented/` 按 6 个视觉域组织增强图片。每个 domain 下通常包含裁剪增强、高分辨率处理和拼接增强三类图片：

```text
augmented/
  General/
    cropped/                   # 裁剪增强
    high_resolution/           # 高分辨率处理
    stitched/                  # 拼接增强
  Remote_sensing/
  Histopathology/
  Cellular_Microscopy/
  Microbiology/
  Agriculture/
```

`tools/` 中根目录保留总处理脚本，子目录保存被总脚本调用的具体处理脚本：

```text
tools/
  convert_all_sources.py              # 统一格式转换入口
  rebuild_restricted_derived_images.py
                                      # 统一增强图片重建入口
  audit_annotation_image_paths.py      # 统一路径审计入口
  converters/                          # 各数据集格式转换子脚本
  rebuilders/                          # 裁剪、拼接、高分辨率增强图片重建子脚本
  auditors/                            # 路径、结构和闭包检查子脚本
  downloaders/                         # 少量辅助下载脚本，例如 Objects365 patch 下载
  recipes/                             # 增强图片重建参数
```

运行数据集准备、增强图片重建和最终审计命令时，请将本目录作为数据集根目录。

## 原始数据集准备

所有原始数据集都应组织在 `images/{dataset_name}/` 下。每个数据集的下载压缩包保留在自己的文件夹中，并在同一文件夹内原地解压。这样 CLOC 数据集 JSON 中的相对路径可以直接从 `data/` 根目录解析。

下面每个数据集小节都会给出三类关键信息。

- 第一，推荐下载地址。强烈建议使用本文给出的入口下载原始数据集，因为不同下载地址或镜像得到的数据集可能存在文件夹结构、图片命名甚至图片数量差异；本文中的下载入口已在 2026 年 5 月重新确认，并用于当前 CLOC 数据集的本地验证。通常情况下，点击本文提供的下载链接后，不需要再自行寻找其他跳转页面，数据集下载入口就在打开后的页面或链接中。
- 第二，下载后的压缩包应如何放置。大多数情况下，只需要把压缩包放到对应的 `images/{dataset_name}/` 文件夹下。
- 第三，原地解压后的期望文件夹结构。CLOC 数据集标注、转换脚本和重建脚本都假定图片路径与该结构一致；如果使用其他来源或手动改动目录结构，可能导致标注无法索引图片或处理脚本失败。只要使用本文推荐入口下载，并将压缩包放到对应数据集目录中原地解压，通常即可得到下方列出的期望结构。

用户只需要准备这些源数据集的图片文件；不需要下载或使用它们的原始标注文件。CLOC 数据集统一使用 `annotations/` 中提供的 CLOC 统一标注。

### AGAR

AGAR 数据集可从 [OpenDataLab AGAR 页面](https://opendatalab.com/OpenDataLab/AGAR/tree/main/raw) 下载。将下载得到的压缩包放到 `images/AGAR/`，然后在该目录下原地解压，不要展平或重命名解压出的文件夹。

解压前：

```text
data/
  images/
    AGAR/
      AGAR.tar.gz
```

例如，原地解压命令如下：

```bash
cd /path/to/data/images/AGAR
tar -xzf AGAR.tar.gz
```

解压后，期望结构开头为：

```text
AGAR/
  AGAR.tar.gz
  AGAR_id_list.md
  dataset/
    1.jpg
    1.json
    2.jpg
    2.json
    ...
```

### BCData

BCData 可从 [官方页面](https://sites.google.com/view/bcdataset) 下载。将下载得到的压缩包放到 `images/BCData/`，然后在原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    BCData/
      BCData.zip
```

解压后期望结构为：

```text
BCData/
  BCData.zip
  BCData/
    images/
      train/
        *.png
      validation/
        *.png
      test/
        *.png
    annotations/
      train/
      validation/
      test/
```

### BriFiSeg

BriFiSeg 可从 [Zenodo](https://zenodo.org/records/7195636) 下载。请注意，下载 `3channels.tar`，而不是 `1channel.tar`。CLOC 使用三通道 brightfield 输入；`1channel.tar` 不足以复现 CLOC 数据集图片。将 `3channels.tar` 放到 `images/BriFiSeg/`，并使用下面给出的命令解压外层 tar 和内部 zip。

解压前：

```text
data/
  images/
    BriFiSeg/
      3channels.tar
```

可以使用下面的命令解压外层 `3channels.tar` 和内部的 `Task*.zip`：

```bash
cd data/images/BriFiSeg
tar -xf 3channels.tar
for z in 3channels/Task*.zip; do
  name="$(basename "$z" .zip)"
  mkdir -p "$name"
  unzip -q "$z" -d "$name"
done
```

BriFiSeg 还需要进行图片格式转换：CLOC 数据集 JSON 引用的是 `unified_images/*.png`，而官方 `3channels.tar` 中提供的是 NIfTI 格式的三通道 brightfield 输入。下面的数据集专用命令会使用 `nibabel` 读取 `*_0000.nii.gz`、`*_0001.nii.gz` 和 `*_0002.nii.gz`，将三通道堆叠并归一化为 RGB PNG。如果尚未手动解压，该脚本也会自动完成上述解压步骤；如果已经解压，则会直接复用现有目录。

您可以立即运行下面的 BriFiSeg 专用转换命令；也可以等所有源数据集都下载并解压完成后，在文档末尾的“转换阶段”统一运行 `python tools/convert_all_sources.py --run`。这两种工作流是等价的，因为统一转换脚本只是按 manifest 依次调用各数据集自己的转换脚本。后续需要格式转换的数据集仍会给出单独转换命令，但不再重复说明统一转换流程。

```bash
cd data
python -m pip install nibabel
python tools/converters/convert_brifiseg_3channels_to_unified_images.py --workspace . --overwrite
```

转换后期望结构为：

```text
BriFiSeg/
  3channels.tar
  3channels/
    Task006_A549.zip
    Task012_HELA.zip
    ...
  Task006_A549/
    Task006_A549/
      imagesTr/
      imagesTs/
  Task012_HELA/
  ...
  unified_images/
    *.png
```

### CARPK / PUCPR+

CARPK 和 PUCPR+ 可从 [LPN 项目页面](https://lafi.github.io/LPN/) 下载。下载页面提供的是合并加密压缩包；该压缩包需要解压密码。请先在上述下载页面填写作者要求的数据集申请/EULA 表格；在您填写完表单之后，解压密码通常会立刻发送到您的邮箱。收到密码后，将下载得到的压缩包放到 `images/CARPK_PUCPR/`，并在该目录下原地解压。

解压前：

```text
data/
  images/
    CARPK_PUCPR/
      datasets.zip
```

解压后期望结构为：

```text
CARPK_PUCPR/
  datasets.zip
  datasets/
    CARPK_devkit/
      data/
        Images/
          *.png
        Annotations/
        ImageSets/
    PUCPR+_devkit/
      data/
        Images/
          *.jpg
        Annotations/
        ImageSets/
    tool/
```

### CellBinDB

CellBinDB 可从 [Zenodo 数据页面](https://zenodo.org/records/15370205) 下载。CLOC 数据集只需要其中的 `CellBinDB.zip`。下载后将 `CellBinDB.zip` 放到 `images/CellBinDB/`，然后原地解压。

解压前：

```text
data/
  images/
    CellBinDB/
      CellBinDB.zip
```

解压后源数据结构为：

```text
CellBinDB/
  CellBinDB.zip
  CellBinDB/
    <platform>/
      <sample_name>/
        <sample_name>-img.tif
        <sample_name>-mask.tif
        <sample_name>-instancemask.tif
```

解压后，因为 CellBinDB 原始图片是 TIFF 格式，还需要执行 TIFF 到 PNG 的格式转换。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本。

```bash
python tools/converters/convert_cellbindb_tif_to_images_png.py --verify --report metadata/cellbindb_tif_to_png_full_report.json
```

该步骤会生成：

```text
CellBinDB/
  images_png/
    <sample_name>-img.png
```

### Cityscapes

Cityscapes 的 left 8-bit 图片包可从 [Cityscapes 下载页面](https://www.cityscapes-dataset.com/login/) 下载。下载前需要先在 Cityscapes 官网注册账号并登录。下载 `leftImg8bit_trainvaltest.zip` 后，将其放到 `images/Cityscapes/`，并原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    Cityscapes/
      leftImg8bit_trainvaltest.zip
```

解压后期望结构为：

```text
Cityscapes/
  leftImg8bit_trainvaltest.zip
  leftImg8bit/
    train/
      <city>/
        *_leftImg8bit.png
    val/
      <city>/
        *_leftImg8bit.png
    test/
      <city>/
        *_leftImg8bit.png
```

### CoNIC

CoNIC 可从 [Kaggle](https://www.kaggle.com/api/v1/datasets/download/aadimator/conic-challenge-dataset) 下载。将下载得到的压缩包放到 `images/CoNIC/`，然后原地解压。

解压前：

```text
data/
  images/
    CoNIC/
      archive.zip
```

解压后源数据结构为：

```text
CoNIC/
  archive.zip
  data/
    images.npy
    labels.npy
    patch_info.csv
    counts.csv
```

解压后，CoNIC 还需要执行 NPY 到 PNG 的准备步骤。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_conic_npy_to_images_png.py --verify --report metadata/conic_npy_to_png_full_report.json
```

该步骤会生成：

```text
CoNIC/
  images_png/
    <patch_name>.png
```

### DeepBacs

CLOC 使用的 DeepBacs 图片来自官方 Zenodo 记录。DeepBacs 被拆成多个任务包，因此需要分别下载下面 6 个 `DeepBacs_Data_*.zip` 文件；不需要下载页面中的 model zip 或示例图片。

- [E. coli brightfield segmentation](https://zenodo.org/records/5550935)：`DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip`
- [B. subtilis fluorescence segmentation](https://zenodo.org/records/5550968)：`DeepBacs_Data_Segmentation_B.subtilis_FtsZ_dataset.zip`
- [S. aureus widefield segmentation](https://zenodo.org/records/5550933)：`DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip`
- [Mixed segmentation / StarDist](https://zenodo.org/records/5551009)：`DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip`
- [E. coli growth stage object detection](https://zenodo.org/records/5551016)：`DeepBacs_Data_Object_Detection_E.coli_Growth_Stage.zip`
- [E. coli antibiotic phenotyping object detection](https://zenodo.org/records/5551057)：`DeepBacs_Data_Object_Detection_E.coli_Antibiotic_Phenotyping.zip`

将这 6 个压缩包都放到 `images/DeepBacs/`，并使用下面给出的命令批量解压。

解压前：

```text
data/
  images/
    DeepBacs/
      DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip
      DeepBacs_Data_Segmentation_B.subtilis_FtsZ_dataset.zip
      DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip
      DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip
      DeepBacs_Data_Object_Detection_E.coli_Growth_Stage.zip
      DeepBacs_Data_Object_Detection_E.coli_Antibiotic_Phenotyping.zip
```

可以使用下面的命令解压这 6 个任务压缩包：

```bash
cd data/images/DeepBacs
for z in DeepBacs_Data_*.zip; do
  unzip -q -o "$z"
done
```

解压后源数据结构开头为：

```text
DeepBacs/
  train/
    brightfield/
    masks_RoiMap/
  test/
    brightfield/
    masks_RoiMap/
  StarDist_dataset/
    test/
      fluorescence/
      masks/
  brightfield_dataset/
    train/full_images/brightfield/
    test/brightfield/
  fluorescence_dataset/
    train/full_images/fluorescence/
    test/fluorescence/
```

解压后，DeepBacs 还需要执行 TIFF 到 PNG 的准备步骤。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_deepbacs_tif_to_pngimages.py --overwrite --verify-output --report metadata/deepbacs_tif_to_pngimages_full_report.json
```

该步骤会生成：

```text
DeepBacs/
  PNGImages/
    DeepBacs_<subset>_<split>_<name>.png
```

### DIOR

DIOR 可从 [OpenDataLab](https://opendatalab.com/OpenDataLab/DIOR/tree/main/raw) 下载。将下载得到的压缩包放到 `images/DIOR/`，然后原地解压。

解压前：

```text
data/
  images/
    DIOR/
      DIOR.tar.gz
```

解压后期望源数据结构为：

```text
DIOR/
  DIOR.tar.gz
  DIOR/
    Annotations/
      Horizontal Bounding Boxes/
      Oriented Bounding Boxes/
    ImageSets/
    JPEGImages-trainval/
      *.jpg
    JPEGImages-test/
      *.jpg
```

### DOTAv1.0

DOTAv1.0 可从 [DOTA 数据集页面](https://captain-whu.github.io/DOTA/dataset.html) 下载。CLOC 只需要下载 DOTAv1.0 的 Training set 和 Validation set，不需要下载 Testing images。将下载得到的 train 和 val 文件夹组织到 `images/DOTAv1.0/` 下。

解压内部 image part 前：

```text
data/
  images/
    DOTAv1.0/
      train/
        images/
          part1.zip
          part2.zip
          ...
      val/
        images/
          part1.zip
          part2.zip
```

每个 image part 应解压为以 zip 名命名的目录。有些 DOTA zip 内部包含一层 `images/`，有些直接包含 PNG，因此建议使用 `unzip -j` 将每个 zip 展平到自己的 `partN/` 目录：

```bash
cd data/images/DOTAv1.0
for z in train/images/part*.zip val/images/part*.zip; do
  part="$(basename "$z" .zip)"
  mkdir -p "$(dirname "$z")/$part"
  unzip -q -o -j "$z" -d "$(dirname "$z")/$part"
done
```

解压后期望结构包括：

```text
DOTAv1.0/
  train/
    images/
      part1/
        P0005.png
      part2/
      ...
  val/
    images/
      part1/
      part2/
```

### DroneCrowd

DroneCrowd 可从 [OpenDataLab](https://opendatalab.com/OpenDataLab/DroneCrowd/tree/main/raw) 下载。将下载得到的 `DroneCrowd.tar.gz.00` 放到 `images/DroneCrowd/`，并使用下面给出的命令解压外层压缩包和其中的内部数据 zip。

解压前：

```text
data/
  images/
    DroneCrowd/
      DroneCrowd.tar.gz.00
```

可以使用下面的命令解压外层压缩包和内部的 `train_data.zip`、`val_data.zip`、`test_data.zip`：

```bash
cd data/images/DroneCrowd
tar -xzf DroneCrowd.tar.gz.00
cd DroneCrowd
for z in train_data.zip val_data.zip test_data.zip; do
  unzip -q -o "$z"
done
```

解压后期望结构为：

```text
DroneCrowd/
  DroneCrowd.tar.gz.00
  DroneCrowd/
    README.md
    annotations.zip
    train_data.zip
    val_data.zip
    ...
    train_data/
      images/
        *.jpg
      ground_truth/
    val_data/
      ...
    test_data/
      ...
```

### EndoNuke

EndoNuke 可从 [EndoNuke 官方页面](https://endonuke.ispras.ru/) 下载，页面中的下载入口会指向 `data.zip`。将该压缩包放到 `images/EndoNuke/`，然后原地解压，不要展平或重命名解压出的文件夹。

解压前：

```text
data/
  images/
    EndoNuke/
      data.zip
```

解压后期望结构开头为：

```text
EndoNuke/
  data/
dataset/
  files_lists/
  images/
  images_context/
  labels/
  metadata/
master_ymls/
  agreement.yaml
  bulk.yaml
  everything.yaml
  hidden_agreement.yaml
  posterior_agreement.yaml
  preliminary_agreement.yaml
  unique.yaml
  data.zip
```

### FSC147

FSC147 可从 [Kaggle](https://www.kaggle.com/api/v1/datasets/download/xuncngng/fsc147-0) 下载。将下载得到的压缩包放到 `images/FSC147/`，然后原地解压。

解压前：

```text
data/
  images/
    FSC147/
      archive.zip
```

解压后期望结构为：

```text
FSC147/
  archive.zip
  FSC147/
    ImageClasses_FSC147.txt
    Train_Test_Val_FSC_147.json
    annotation_FSC147_384.json
    images_384_VarV2/
      *.jpg
    gt_density_map_adaptive_384_VarV2/
      *.npy
```

### FSCD-LVIS

FSCD-LVIS 可从 [OpenDataLab](https://opendatalab.com/OpenDataLab/FSCD-LVIS/tree/main/raw) 下载。将下载得到的 `FSCD-LVIS.tar.gz` 放到 `images/FSCD_LVIS/`，并使用下面给出的命令解压外层压缩包和内部的 `FSCD_LVIS.zip`。

解压前：

```text
data/
  images/
    FSCD_LVIS/
      FSCD-LVIS.tar.gz
```

可以使用下面的命令解压外层压缩包和内部的 `FSCD_LVIS.zip`：

```bash
cd data/images/FSCD_LVIS
tar -xzf FSCD-LVIS.tar.gz
cd FSCD-LVIS
unzip -q -o FSCD_LVIS.zip
```

解压后期望结构为：

```text
FSCD_LVIS/
  FSCD-LVIS.tar.gz
  FSCD-LVIS/
    FSCD_147.zip
    FSCD_LVIS.zip
    Readme.MD
    FSCD_LVIS/
      annotations/
      images/
        *.jpg
        train/
        test/
      masks/
```

### GWHD_2021

GWHD_2021 可从 [Kaggle](https://www.kaggle.com/datasets/vbookshelf/global-wheat-head-dataset-2021) 下载。将下载得到的压缩包放到 `images/GWHD_2021/`，然后原地解压，不要展平或重命名解压出的文件夹。

解压前：

```text
data/
  images/
    GWHD_2021/
      archive.zip
```

解压后期望结构开头为：

```text
GWHD_2021/
  gwhd_2021/
    competition_train.csv
    competition_val.csv
    competition_test.csv
    images/
      0007634580386bd39d4d0d24df58893c3bb967e12d6fc065ce8659e9acacc928.png
      00319488e879a811698174d9f26ef174f2f108a13e12edee5a3c50899ed26336.png
      ...
```

### JHU-CROWD++

JHU-CROWD++ 可从 [官方网站](https://www.crowd-counting.com/) 下载。将下载得到的官方压缩包放到 `images/JHU-CROWD++/`，然后原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    JHU-CROWD++/
      jhu_crowd_v2.0.zip
```

解压后期望结构为：

```text
JHU-CROWD++/
  jhu_crowd_v2.0.zip
  jhu_crowd_v2.0/
    train/
      images/
        *.jpg
      gt/
    val/
      images/
        *.jpg
      gt/
    test/
      images/
        *.jpg
      gt/
```

### LIVECell

LIVECell 图片压缩包可从 [LIVECell 项目页面](https://sartorius-research.github.io/LIVECell/) 下载。将下载得到的图片压缩包放到 `images/LIVECELL/`，然后原地解压。

解压前：

```text
data/
  images/
    LIVECELL/
      images.zip
```

解压后源数据结构为：

```text
LIVECELL/
  images.zip
  images/
    livecell_train_val_images/
      *.tif
    livecell_test_images/
      *.tif
```

解压后，LIVECell 还需要执行 TIFF 到 PNG 的准备步骤。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_livecell_tif_to_images_png.py --overwrite --verify --report metadata/livecell_tif_to_images_png_full_report.json
```

该步骤会生成：

```text
LIVECELL/
  images_png/
    *.png
```

### Lizard

Lizard 可从 [Kaggle](https://www.kaggle.com/api/v1/datasets/download/aadimator/lizard-dataset) 下载。将下载得到的压缩包放到 `images/Lizard/`，然后原地解压，不要展平或重命名解压出的文件夹。

解压前：

```text
data/
  images/
    Lizard/
      archive.zip
```

解压后期望结构开头为：

```text
Lizard/
  archive.zip
  lizard_images1/
    Lizard_Images1/
      consep_1.png
      consep_2.png
      ...
  lizard_images2/
    Lizard_Images2/
      ...
  lizard_labels/
    ...
```

### MoNuSAC

MoNuSAC 可从 [官方挑战页面](https://monusac-2020.grand-challenge.org/Data/) 下载。CLOC 只需要下载页面中的 Training Data；将下载得到的压缩包放到 `images/MoNuSAC/`，然后原地解压。

解压前：

```text
data/
  images/
    MoNuSAC/
      MoNuSAC_images_and_annotations.zip
```

解压后原始结构为：

```text
data/
  images/
    MoNuSAC/
      MoNuSAC_images_and_annotations.zip
      MoNuSAC_images_and_annotations/
        TCGA-*/
          *.tif
          *.xml
          *.svs
```

MoNuSAC 还需要从 TIFF 文件生成 PNG 图片。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_monusac_tif_to_png_images.py
```

转换后期望结构为：

```text
MoNuSAC/
  png_images/
    *.png
```


### MoNuSeg

MoNuSeg 可从 [官方挑战页面](https://monuseg.grand-challenge.org/Data/) 下载。下载 `MoNuSeg 2018 Training Data.zip` 和 `MoNuSegTestData.zip`，都放到 `images/MoNuSeg/`，然后分别原地解压。

解压前：

```text
data/
  images/
    MoNuSeg/
      MoNuSeg 2018 Training Data.zip
      MoNuSegTestData.zip
```

解压后原始结构为：

```text
data/
  images/
    MoNuSeg/
      MoNuSeg 2018 Training Data.zip
      MoNuSegTestData.zip
      MoNuSeg 2018 Training Data/
        Tissue Images/
          *.tif
        Annotations/
          *.xml
      MoNuSegTestData/
        *.tif
        *.xml
```

MoNuSeg 还需要从官方 TIFF 文件生成 PNG 图片。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_monuseg_tif_to_png.py --verify
```

转换后期望引用的 PNG 路径为：

```text
data/
  images/
    MoNuSeg/
      MoNuSeg 2018 Training Data/
        Tissue Images/
          *.png
      MoNuSegTestData/
        *.png
```


### MTC

Maize Tassel Counting 数据集可从 [MTC GitHub 页面](https://github.com/poppinace/mtc) 下载。将下载得到的压缩包放到 `images/MTC/`，然后原地解压。

解压前：

```text
data/
  images/
    MTC/
      Maize Tassel Counting Dataset.zip
```

解压后期望结构为：

```text
MTC/
  Maize Tassel Counting Dataset.zip
  Maize Tassel Counting Dataset/
    Gucheng2012/
      Images/
        *.jpg
      Annotations/
        *.mat
    Gucheng2014/
      Images/
      Annotations/
    Taian2010_1/
      Images/
      Annotations/
    ...
```

### NuCLS

NuCLS 可从 [OpenDataLab](https://opendatalab.com/OpenDataLab/NuCLS/tree/main/raw) 下载。将下载得到的压缩包放到 `images/NuCLS/`，然后原地解压，不要展平解压出的 `NuCLS/` 文件夹。

解压前：

```text
data/
  images/
    NuCLS/
      NuCLS.tar.gz.00
```

解压后期望结构为：

```text
NuCLS/
  NuCLS.tar.gz.00
  NuCLS/
    BootstrapControl/
    EvaluationSet/
    QC/
      csv/
      mask/
      rgb/
        *.png
      train_test_splits/
      visualization/
    UnbiasedControl/
    noQC/
```

### NuInsSeg

NuInsSeg 可从 [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10518968) 下载。将下载得到的压缩包放到 `images/NuInsSeg/`，然后原地解压，不要展平解压出的 organ 文件夹。

解压前：

```text
data/
  images/
    NuInsSeg/
      NuInsSeg.zip
```

解压后期望结构为：

```text
NuInsSeg/
  NuInsSeg.zip
  human bladder/
    tissue images/
      *.png
    label masks/
      *.tif
  human brain/
    tissue images/
      *.png
    label masks/
      *.tif
  ...
```


### NWPU-CROWD

NWPU-CROWD 可从 [NWPU-CROWD 官网](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/) 下载。CLOC 只需要下载 5 个图片压缩包：`images_part1.zip` 到 `images_part5.zip`。将这 5 个压缩包放在 `images/NWPU-CROWD/` 下，并使用下面给出的命令解压到同一个图片文件夹。

解压前：

```text
data/
  images/
    NWPU-CROWD/
      images_part1.zip
      images_part2.zip
      ...
      images_part5.zip
```

将五个 image-part 压缩包解压到同一个 `NWPU-CROWD/` 图片文件夹：

```bash
cd data/images/NWPU-CROWD
mkdir -p NWPU-CROWD
for z in images_part*.zip; do
  unzip -q -o "$z" -d NWPU-CROWD
done
```

解压后期望结构为：

```text
NWPU-CROWD/
  images_part1.zip
  ...
  images_part5.zip
  NWPU-CROWD/
    0001.jpg
    0002.jpg
    ...
    5109.jpg
```

### NWPU-MOC

NWPU-MOC 可从 [官方 GitHub 页面](https://github.com/lyongo/NWPU-MOC) 下载。将下载得到的压缩包放到 `images/NWPU-MOC/`，然后原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    NWPU-MOC/
      NWPU-MOC.zip
```

解压后期望结构为：

```text
NWPU-MOC/
  NWPU-MOC.zip
  NWPU-MOC/
    rgb/
      *.png
    ir/
      *.png
    gt/
      *.npz
    gt14/
    annotations/
```

### NWPU-VHR-10

NWPU-VHR-10 可从 [Kaggle](https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10) 下载。将下载得到的压缩包放到 `images/NWPU-VHR-10/`，然后原地解压。

解压前：

```text
data/
  images/
    NWPU-VHR-10/
      archive.zip
```

解压后期望结构为：

```text
NWPU-VHR-10/
  archive.zip
  NWPU VHR-10 dataset/
    positive image set/
      *.jpg
    negative image set/
      *.jpg
    ground truth/
      *.txt
```

### Objects365-2020

Objects365-2020 使用官方 KS3 patch 文件下载。注意，这里的 base URL 不是可浏览网页入口，直接在浏览器中打开可能返回 404；请使用下面的脚本下载具体 patch 文件。该数据集较大，全部 patch 的压缩下载量约为 368 GB，解压后会占用更多磁盘空间。

运行脚本后会自动下载并解压全部所需 patch 文件：

```bash
cd data
bash tools/downloaders/download_objects365_patches_no_proxy.sh
```

解压后期望结构为：

```text
data/
  images/
    Objects365-2020/
      train/
        patch0/
          objects365_v1_*.jpg
        ...
        patch50/
          objects365_v1_*.jpg
      val/
        images/
          v1/
            patch0/
              objects365_v1_*.jpg
            ...
            patch15/
              objects365_v1_*.jpg
          v2/
            patch16/
              objects365_v2_*.jpg
            ...
            patch43/
              objects365_v2_*.jpg
```

### Rebar Counting

Rebar Counting 可从 [Roboflow](https://universe.roboflow.com/search?q=rebar%20counting) 下载。请搜索名为 `rebar counting`、作者为 `fyp2`、约 250 张图片的数据集，并导出 COCO 格式压缩包。将下载得到的压缩包放到 `images/rebar_counting/`，然后原地解压。

解压前：

```text
data/
  images/
    rebar_counting/
      rebar counting.coco.zip
```

解压后期望结构为：

```text
rebar_counting/
  rebar counting.coco.zip
  train/
    *.jpg
    _annotations.coco.json
```

### RSOD

RSOD 可从 [OpenDataLab](https://opendatalab.com/OpenDataLab/RSOD/tree/main/raw) 下载。将下载得到的 `RSOD.tar.gz` 放到 `images/RSOD/`，并使用下面给出的命令解压外层压缩包和四个类别内部 zip。

解压前：

```text
data/
  images/
    RSOD/
      RSOD.tar.gz
```

解压外层压缩包后：

```text
data/
  images/
    RSOD/
      RSOD.tar.gz
      RSOD/
        aircraft.zip
        oiltank.zip
        overpass.zip
        playground.zip
```

解压四个内部压缩包：

```bash
cd data/images/RSOD/RSOD
for z in aircraft.zip oiltank.zip overpass.zip playground.zip; do
  unzip -q -o "$z"
done
```

解压后期望结构为：

```text
RSOD/
  RSOD/
    aircraft/
      JPEGImages/
        *.jpg
    oiltank/
      JPEGImages/
        *.jpg
    overpass/
      JPEGImages/
        *.jpg
    playground/
      JPEGImages/
        *.jpg
```

### ShanghaiTech

ShanghaiTech crowd counting 数据集可从 [Kaggle](https://www.kaggle.com/api/v1/datasets/download/xyyu18/shanghaitech-crowd-counting-dataset) 下载。将下载得到的压缩包放到 `images/ShanghaiTech/`，然后原地解压。

解压前：

```text
data/
  images/
    ShanghaiTech/
      archive.zip
```

解压后期望结构为：

```text
ShanghaiTech/
  archive.zip
  part_A_final/
    train_data/
      images/
        IMG_*.jpg
      ground_truth/
        GT_IMG_*.mat
    test_data/
      ...
  part_B_final/
    ...
```

### Soybean Pod Images from UAVs

Soybean Pod Images from UAVs 可从 [Kaggle](https://www.kaggle.com/datasets/jiajiali/uav-based-soybean-pod-images) 下载。将下载得到的压缩包放到 `images/soybean_pod/`，然后原地解压。

解压前：

```text
data/
  images/
    soybean_pod/
      archive.zip
```

解压后原始结构为：

```text
data/
  images/
    soybean_pod/
      archive.zip
      dataset/
        *.bmp
        *.json
```

Soybean Pod Images from UAVs 还需要从 BMP 文件生成 PNG 图片。可以立即运行下面的数据集专用命令，也可以在文档末尾的“转换阶段”统一运行所有转换脚本：

```bash
python tools/converters/convert_soybean_pod_bmp_to_png.py --verify
```

转换后期望结构为：

```text
soybean_pod/
  dataset_png/
    *.png
```

### UpCount

UpCount 图片可从 [Zenodo](https://zenodo.org/records/12683104/files/images.zip?download=1) 下载。将下载得到的 `images.zip` 放到 `images/upcount/`，然后原地解压。

解压前：

```text
data/
  images/
    upcount/
      images.zip
```

解压后期望结构为：

```text
upcount/
  images.zip
  UP-COUNT/
    images/
      0000/
        *.JPG
      0001/
        *.JPG
      ...
```

### VGG Cell Detection

VGG Cell Detection 可从 [Academic Torrents](https://academictorrents.com/details/b32305598175bb8e03c5f350e962d772a910641c) 下载。将下载得到的压缩包放到 `images/VGG/`，然后解压到 `images/VGG/VGG_cells/`。

解压前：

```text
data/
  images/
    VGG/
      cells.zip
```

解压后期望结构为：

```text
VGG/
  cells.zip
  VGG_cells/
    001cell.png
    001dots.png
    ...
    200cell.png
    200dots.png
```

### VisDrone

VisDrone 可从 [Ultralytics YOLOv5 v1.0 release 页面](https://github.com/ultralytics/yolov5/releases/tag/v1.0) 下载。CLOC 需要下载 `VisDrone2019-DET-train.zip` 和 `VisDrone2019-DET-val.zip`。将这两个压缩包放到 `images/VisDrone/`，然后原地解压，不要展平或重命名解压出的文件夹。

解压前：

```text
data/
  images/
    VisDrone/
      VisDrone2019-DET-train.zip
      VisDrone2019-DET-val.zip
```

解压后期望结构开头为：

```text
VisDrone/
  VisDrone2019-DET-train/
    annotations/
      0000002_00005_d_0000014.txt
      0000002_00448_d_0000015.txt
      ...
    images/
      0000002_00005_d_0000014.jpg
      0000002_00448_d_0000015.jpg
      ...
  VisDrone2019-DET-val/
    annotations/
      ...
    images/
      ...
```

### VOC2007

VOC2007 可从 [PASCAL VOC2007 官方页面](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) 下载。需要下载 `training/validation data` 和 `annotated test data`。将下载得到的两个压缩包都放到 `images/VOCdevkit/`，然后原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    VOCdevkit/
      VOCtrainval_06-Nov-2007.tar
      VOCtest_06-Nov-2007.tar
```

解压后期望结构为：

```text
VOCdevkit/
  VOCtrainval_06-Nov-2007.tar
  VOCtest_06-Nov-2007.tar
  VOCdevkit/
    VOC2007/
      Annotations/
      ImageSets/
      JPEGImages/
        *.jpg
      SegmentationClass/
      SegmentationObject/
```

### xView

xView 可从 [Kaggle](https://www.kaggle.com/datasets/hassanmojab/xview-dataset?resource=download) 下载。将下载得到的压缩包放到 `images/xview/`，然后原地解压，不要展平解压出的文件夹。

解压前：

```text
data/
  images/
    xview/
      archive.zip
```

解压后期望结构为：

```text
xview/
  archive.zip
  train_images/
    train_images/
      *.tif
  val_images/
    val_images/
      *.tif
  train_labels/
    *.geojson
```

## 转换阶段

这一阶段是准备完整 CLOC 数据集时需要完成的步骤之一，除非你已经在前面各数据集小节中手动运行过所有需要的单数据集转换命令。它负责把部分源数据集中的 TIFF、BMP、NPY 或 NIfTI 等非标准图片格式转换为 CLOC 数据集 JSON 实际引用的 PNG 图片。

推荐流程是先预览转换计划，再执行全部转换：

```bash
cd data
python tools/convert_all_sources.py
python tools/convert_all_sources.py --run
```

第一条命令只做 dry-run，会检查各转换步骤所需输入是否存在，并写出 `metadata/conversion_dry_run_report.json`。如果 dry-run 报告缺失输入，请回到对应数据集小节检查下载包和解压结构。第二条命令才会真正生成转换后的图片。

我们还提供一些可选功能，具体如下：

- 如果需要覆盖已经存在的转换结果，可以运行 `python tools/convert_all_sources.py --run --overwrite`。
- 如果只想运行某一个数据集的转换，可以先用 `python tools/convert_all_sources.py --list` 查看 step id，再用 `--only <step_id>` 指定目标步骤。
- 转换计划保存在 `manifests/conversion_manifest.json`，各个数据集专用转换脚本保存在 `tools/converters/`。前文数据集小节中给出的单独转换命令和这里的统一转换命令是等价入口，二者不需要重复执行。

## 重建阶段

这一阶段也是准备完整 CLOC 数据集时需要完成的步骤之一。CLOC 数据集标注引用了一部分增强图片，包括高分辨率切块、裁剪增强图和拼接/马赛克增强图。由于部分增强图片来自受权限限制的源数据集，我们不直接发布这些图片，而是提供重建参数和脚本，让用户在本地从已经下载好的源数据集重新生成。

请在完成原始数据集准备和转换阶段之后，先预览重建计划，再执行重建：

```bash
cd data
python tools/rebuild_restricted_derived_images.py --dry-run --verify
python tools/rebuild_restricted_derived_images.py --verify --overwrite
```

第一条命令不会写入图片，只检查重建参数、源图片和目标路径。第二条命令会将重建结果写入 `augmented/`，不会修改 train/val/test JSON。若 dry-run 报告缺失源图片或重建参数错误，请先修正对应源数据集目录。


## 最终审计阶段

最终审计是完整准备流程的最后一步，必须在原始数据集解压、转换和重建都完成后运行。它只检查路径，不生成新图片，也不修改标注。审计目标是确认 CLOC 数据集 train/val/test 划分 JSON 中的每一个 `image_path` 都能在当前 `data/` 工作区内找到对应图片。

运行审计：

```bash
cd data
python tools/audit_annotation_image_paths.py \
  --workspace . \
  --report metadata/annotation_image_path_audit_final.json \
  --markdown metadata/annotation_image_path_audit_final.md
```

审计默认扫描未按类别展开的 split 文件：

```text
annotations/train_split.json
annotations/val_split.json
annotations/test_split.json
```

审计通过的标准是所有 split 的 `missing` 都等于 0。如果任意 split 报告 `missing > 0`，说明仍有标注路径无法解析，CLOC 数据集还没有准备完成。此时请优先查看 `metadata/annotation_image_path_audit_final.md`，其中会按数据集根路径汇总缺失路径，并给出部分缺失样本示例用于定位问题。
