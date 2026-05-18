# CLOC Dataset Release

<p align="right">
  <a href="README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

This document explains how to prepare the complete CLOC dataset. We provide all CLOC annotation files and a subset of directly distributable augmented images. Users need to prepare the raw images of each source dataset according to this document, then run the unified conversion, rebuild, and audit scripts. After these steps, the resulting workspace can be fully indexed by the CLOC dataset annotations.

## CLOC Annotation and Augmented-Image Archives

First, download the CLOC annotation archive and the directly distributable augmented-image archive from the following links:

- [Google Drive](https://drive.google.com/drive/folders/1jt_CLqfQRVV9Vzll2Aqlf1tQNbWbezSe?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1cCFxHLcU6o1EFGjtxZgseg?pwd=cloc)

Download `cloc_annotations.zip` and `cloc_augmented.zip`, then place both archives under the `data/` root:

```text
data/
  cloc_annotations.zip
  cloc_augmented.zip
```

Then extract them from the `data/` root:

```bash
cd /path/to/data
unzip -q cloc_annotations.zip
unzip -q cloc_augmented.zip
```

After extraction, the following folders should be available:

```text
data/
  annotations/
  augmented/
```

## Layout

```text
data/
  annotations/        # CLOC dataset JSONs with data-root-relative image paths
  tools/              # entry scripts plus subfolders; root keeps only conversion, rebuild, and audit entry points
  metadata/           # script reports and audit results
  images/             # source dataset archives/extracted folders, organized by dataset name
  augmented/          # augmented images referenced by CLOC annotations
```

The main contents of `annotations/` are:

```text
annotations/
  *_split.json                 # original train/val/test splits, organized by image record
  *_expanded_by_class.json     # split records expanded so each annotation record contains one target category
  class_group.json             # category group split
  seen_unseen_split.json       # seen/unseen category split
```

The `augmented/` folder is organized by the six visual domains. Each domain usually contains three types of augmented images: cropped augmentation, high-resolution processing, and stitched augmentation:

```text
augmented/
  General/
    cropped/                   # cropped augmentation
    high_resolution/           # high-resolution processing
    stitched/                  # stitched augmentation
  Remote_sensing/
  Histopathology/
  Cellular_Microscopy/
  Microbiology/
  Agriculture/
```

The `tools/` root keeps the main entry scripts, while subfolders contain the
dataset-specific scripts called by those entries:

```text
tools/
  convert_all_sources.py              # unified source-image conversion entry
  rebuild_restricted_derived_images.py
                                      # unified augmented-image rebuild entry
  audit_annotation_image_paths.py      # unified path-audit entry
  converters/                          # per-dataset format-conversion scripts
  rebuilders/                          # crop, stitch, and high-resolution rebuild scripts
  auditors/                            # path, layout, and closure-check scripts
  downloaders/                         # small helper download scripts, e.g. Objects365 patches
  recipes/                             # augmented-image reconstruction parameters
```

Use this directory as the dataset root when running the preparation, rebuild, and audit commands.

## Source Dataset Setup

All source datasets should be organized under `images/{dataset_name}/`. Each dataset keeps its downloaded archive in its own folder, and the archive should be extracted in place under that same folder so that relative paths in the CLOC dataset JSON files can be resolved from the `data/` root.

Each dataset section below provides three key pieces of information.

- First, the recommended download entry. We strongly recommend using the entries listed here because different mirrors or download routes may produce different folder layouts, image names, or even image counts. The entries in this document were re-checked in May 2026 and used to validate the current CLOC dataset. In most cases, after clicking the download links provided here, you do not need to search for additional redirect pages; the dataset download entry is on the opened page or link.
- Second, where to place the downloaded archive. In most cases, the archive only needs to be placed under the corresponding `images/{dataset_name}/` folder.
- Third, the expected folder structure after in-place extraction. The CLOC dataset annotations, conversion scripts, and rebuild scripts assume that image paths follow this structure. If you use another source or manually change the extracted layout, annotation paths or processing scripts may fail. If you download from the recommended entry and extract the archive in place under the dataset folder, the expected structure below should normally be obtained directly.

Users only need to prepare the image files from these source datasets; the original annotations of the source datasets are not needed. CLOC uses the unified CLOC annotations provided under `annotations/`.

### AGAR

The AGAR dataset can be downloaded from the [OpenDataLab AGAR page](https://opendatalab.com/OpenDataLab/AGAR/tree/main/raw). Save the downloaded archive(s) under `images/AGAR/`, then extract them in place without flattening or renaming the extracted folders.

Before extraction:

```text
data/
  images/
    AGAR/
      AGAR.tar.gz
```

For example, extract in place with:

```bash
cd /path/to/data/images/AGAR
tar -xzf AGAR.tar.gz
```

After extraction, the expected structure begins as:

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

The BCData dataset can be downloaded from the [official BCData page](https://sites.google.com/view/bcdataset). Place the downloaded archive under `images/BCData/`, then extract it in place without flattening the extracted folder.

Before extraction:

```text
data/
  images/
    BCData/
      BCData.zip
```

After extraction, the expected structure is:

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

The BriFiSeg dataset can be downloaded from [Zenodo](https://zenodo.org/records/7195636). Please download `3channels.tar`, not `1channel.tar`. CLOC uses the three-channel brightfield input; `1channel.tar` is not sufficient for reproducing the CLOC dataset images. Place `3channels.tar` under `images/BriFiSeg/`, then use the command below to extract the outer tar archive and the inner zip files.

Before extraction:

```text
data/
  images/
    BriFiSeg/
      3channels.tar
```

You can extract the outer `3channels.tar` archive and the inner `Task*.zip` files with:

```bash
cd data/images/BriFiSeg
tar -xf 3channels.tar
for z in 3channels/Task*.zip; do
  name="$(basename "$z" .zip)"
  mkdir -p "$name"
  unzip -q "$z" -d "$name"
done
```

BriFiSeg also requires an image-format conversion: the CLOC dataset JSON references `unified_images/*.png`, while the official `3channels.tar` provides three-channel brightfield inputs in NIfTI format. The dataset-specific command below uses `nibabel` to read `*_0000.nii.gz`, `*_0001.nii.gz`, and `*_0002.nii.gz`, stacks the three channels, and normalizes them into RGB PNG images. If the archives have not been extracted manually, the script will perform the same extraction steps automatically; if they are already extracted, it reuses the existing directories.

You can run the BriFiSeg-specific conversion command immediately, or wait until all source datasets are downloaded and extracted and then run the unified conversion command in the final "Conversion Stage": `python tools/convert_all_sources.py --run`. The two workflows are equivalent because the unified conversion script simply invokes the dataset-specific converters listed in the manifest. Later dataset sections still provide the individual conversion commands, but do not repeat this unified-conversion note.

```bash
cd data
python -m pip install nibabel
python tools/converters/convert_brifiseg_3channels_to_unified_images.py --workspace . --overwrite
```

After conversion, the expected structure is:

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

The CARPK and PUCPR+ datasets can be downloaded from the [LPN project page](https://lafi.github.io/LPN/). The download page provides a combined encrypted archive, and this archive requires an extraction password. Fill in the dataset request/EULA form on the download page first; after you submit the form, the extraction password is usually sent to your email immediately. After receiving the password, place the downloaded archive under `images/CARPK_PUCPR/`, then extract it in place under that folder.

Before extraction:

```text
data/
  images/
    CARPK_PUCPR/
      datasets.zip
```

After extraction, the expected structure is:

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

The CellBinDB dataset can be downloaded from the [Zenodo dataset page](https://zenodo.org/records/15370205). CLOC only needs `CellBinDB.zip` from that record. Place `CellBinDB.zip` under `images/CellBinDB/`, then extract it in place.

Before extraction:

```text
data/
  images/
    CellBinDB/
      CellBinDB.zip
```

After extraction, the expected source structure is:

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

After extraction, CellBinDB still needs a TIFF-to-PNG format conversion because its source images are stored as TIFF files. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage".

```bash
python tools/converters/convert_cellbindb_tif_to_images_png.py --verify --report metadata/cellbindb_tif_to_png_full_report.json
```

This generates:

```text
CellBinDB/
  images_png/
    <sample_name>-img.png
```

### Cityscapes

The Cityscapes left 8-bit image package can be downloaded from the [Cityscapes download page](https://www.cityscapes-dataset.com/login/). You need to register a Cityscapes account and log in before downloading. Download `leftImg8bit_trainvaltest.zip`, place it under `images/Cityscapes/`, then extract it in place without flattening the extracted folder.

Before extraction:

```text
data/
  images/
    Cityscapes/
      leftImg8bit_trainvaltest.zip
```

After extraction, the expected structure is:

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

The CoNIC dataset can be downloaded from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/aadimator/conic-challenge-dataset). Place the downloaded archive under `images/CoNIC/`, then extract it in place.

Before extraction:

```text
data/
  images/
    CoNIC/
      archive.zip
```

After extraction, the expected source structure is:

```text
CoNIC/
  archive.zip
  data/
    images.npy
    labels.npy
    patch_info.csv
    counts.csv
```

After extraction, CoNIC also needs an NPY-to-PNG preparation step. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_conic_npy_to_images_png.py --verify --report metadata/conic_npy_to_png_full_report.json
```

This generates:

```text
CoNIC/
  images_png/
    <patch_name>.png
```

### DeepBacs

The DeepBacs images used by CLOC come from the official Zenodo records. DeepBacs is split into multiple task packages, so download the following six `DeepBacs_Data_*.zip` files only; the model zip files and preview images on those pages are not needed.

- [E. coli brightfield segmentation](https://zenodo.org/records/5550935): `DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip`
- [B. subtilis fluorescence segmentation](https://zenodo.org/records/5550968): `DeepBacs_Data_Segmentation_B.subtilis_FtsZ_dataset.zip`
- [S. aureus widefield segmentation](https://zenodo.org/records/5550933): `DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip`
- [Mixed segmentation / StarDist](https://zenodo.org/records/5551009): `DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip`
- [E. coli growth stage object detection](https://zenodo.org/records/5551016): `DeepBacs_Data_Object_Detection_E.coli_Growth_Stage.zip`
- [E. coli antibiotic phenotyping object detection](https://zenodo.org/records/5551057): `DeepBacs_Data_Object_Detection_E.coli_Antibiotic_Phenotyping.zip`

Place all six archives under `images/DeepBacs/`, then use the command below to extract them in batch.

Before extraction:

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

Extract the six task archives with:

```bash
cd data/images/DeepBacs
for z in DeepBacs_Data_*.zip; do
  unzip -q -o "$z"
done
```

After extraction, the expected source structure begins as:

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

After extraction, DeepBacs also needs a TIFF-to-PNG preparation step. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_deepbacs_tif_to_pngimages.py --overwrite --verify-output --report metadata/deepbacs_tif_to_pngimages_full_report.json
```

This generates:

```text
DeepBacs/
  PNGImages/
    DeepBacs_<subset>_<split>_<name>.png
```

### DIOR

The DIOR dataset can be downloaded from [OpenDataLab](https://opendatalab.com/OpenDataLab/DIOR/tree/main/raw). Save the downloaded archive as `DIOR.tar.gz`, place it under `images/DIOR/`, then extract it in place.

Before extraction:

```text
data/
  images/
    DIOR/
      DIOR.tar.gz
```

After extraction, the expected source structure is:

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

The DOTAv1.0 dataset can be downloaded from the [DOTA dataset page](https://captain-whu.github.io/DOTA/dataset.html). CLOC only needs the DOTAv1.0 Training set and Validation set; Testing images are not required. Organize the downloaded train and val folders under `images/DOTAv1.0/`.

Before extracting the nested image parts:

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

Each image part should be extracted as a directory named after its zip. Some DOTA zips contain an inner `images/` folder while others contain PNGs directly, so use `unzip -j` to flatten each zip into its own `partN/` directory:

```bash
cd data/images/DOTAv1.0
for z in train/images/part*.zip val/images/part*.zip; do
  part="$(basename "$z" .zip)"
  mkdir -p "$(dirname "$z")/$part"
  unzip -q -o -j "$z" -d "$(dirname "$z")/$part"
done
```

After extraction, the expected structure includes:

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

The DroneCrowd dataset can be downloaded from [OpenDataLab](https://opendatalab.com/OpenDataLab/DroneCrowd/tree/main/raw). Place the downloaded `DroneCrowd.tar.gz.00` under `images/DroneCrowd/`, then use the command below to extract the outer archive and its inner data zip files.

Before extraction:

```text
data/
  images/
    DroneCrowd/
      DroneCrowd.tar.gz.00
```

Extract the outer archive and the inner `train_data.zip`, `val_data.zip`, and `test_data.zip` archives with:

```bash
cd data/images/DroneCrowd
tar -xzf DroneCrowd.tar.gz.00
cd DroneCrowd
for z in train_data.zip val_data.zip test_data.zip; do
  unzip -q -o "$z"
done
```

After extraction, the expected structure is:

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

The EndoNuke dataset can be downloaded from the [official EndoNuke page](https://endonuke.ispras.ru/); the download entry on that page points to `data.zip`. Save this archive under `images/EndoNuke/`, then extract it in place without flattening or renaming the extracted folders.

Before extraction:

```text
data/
  images/
    EndoNuke/
      data.zip
```

After extraction, the expected structure begins as:

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

The FSC147 dataset can be downloaded from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/xuncngng/fsc147-0). Place the downloaded archive under `images/FSC147/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    FSC147/
      archive.zip
```

After extraction, the expected structure is:

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

The FSCD-LVIS dataset can be downloaded from [OpenDataLab](https://opendatalab.com/OpenDataLab/FSCD-LVIS/tree/main/raw). Place the downloaded `FSCD-LVIS.tar.gz` under `images/FSCD_LVIS/`, then use the command below to extract the outer archive and the inner `FSCD_LVIS.zip` archive.

Before extraction:

```text
data/
  images/
    FSCD_LVIS/
      FSCD-LVIS.tar.gz
```

Extract the outer archive and the inner `FSCD_LVIS.zip` archive with:

```bash
cd data/images/FSCD_LVIS
tar -xzf FSCD-LVIS.tar.gz
cd FSCD-LVIS
unzip -q -o FSCD_LVIS.zip
```

After extraction, the expected structure is:

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

The GWHD_2021 dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/global-wheat-head-dataset-2021). Place the downloaded archive under `images/GWHD_2021/`, then extract it in place without flattening or renaming the extracted folders.

Before extraction:

```text
data/
  images/
    GWHD_2021/
      archive.zip
```

After extraction, the expected structure begins as:

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

The JHU-CROWD++ dataset can be downloaded from the [official JHU-CROWD++ website](https://www.crowd-counting.com/). Save the official archive as `jhu_crowd_v2.0.zip`, place it under `images/JHU-CROWD++/`, then extract it in place without flattening the extracted folder.

Before extraction:

```text
data/
  images/
    JHU-CROWD++/
      jhu_crowd_v2.0.zip
```

After extraction, the expected structure is:

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

The LIVECell image archive can be downloaded from the [LIVECell project page](https://sartorius-research.github.io/LIVECell/). Save the downloaded image archive as `images.zip`, place it under `images/LIVECELL/`, then extract it in place.

Before extraction:

```text
data/
  images/
    LIVECELL/
      images.zip
```

After extraction, the expected source structure is:

```text
LIVECELL/
  images.zip
  images/
    livecell_train_val_images/
      *.tif
    livecell_test_images/
      *.tif
```

After extraction, LIVECell also needs a TIFF-to-PNG preparation step. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_livecell_tif_to_images_png.py --overwrite --verify --report metadata/livecell_tif_to_images_png_full_report.json
```

This generates:

```text
LIVECELL/
  images_png/
    *.png
```

### Lizard

The Lizard dataset can be downloaded from the [Kaggle Lizard dataset archive](https://www.kaggle.com/api/v1/datasets/download/aadimator/lizard-dataset). Place the downloaded archive under `images/Lizard/`, then extract it in place without flattening or renaming the extracted folders.

Before extraction:

```text
data/
  images/
    Lizard/
      archive.zip
```

After extraction, the expected structure begins as:

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

The MoNuSAC dataset can be downloaded from the [official MoNuSAC challenge page](https://monusac-2020.grand-challenge.org/Data/). CLOC only needs the Training Data package on that page. Place the downloaded archive under `images/MoNuSAC/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    MoNuSAC/
      MoNuSAC_images_and_annotations.zip
```

After extraction, the expected raw structure is:

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

MoNuSAC also needs PNG images generated from the TIFF files. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_monusac_tif_to_png_images.py
```

After conversion, the expected structure is:

```text
MoNuSAC/
  png_images/
    *.png
```


### MoNuSeg

The MoNuSeg dataset can be downloaded from the [official MoNuSeg challenge page](https://monuseg.grand-challenge.org/Data/). Download both `MoNuSeg 2018 Training Data.zip` and `MoNuSegTestData.zip`, place them under `images/MoNuSeg/`, then unzip both archives in place.

Before extraction:

```text
data/
  images/
    MoNuSeg/
      MoNuSeg 2018 Training Data.zip
      MoNuSegTestData.zip
```

After extraction, the expected raw structure is:

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

MoNuSeg also needs PNG images generated from the official TIFF files. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_monuseg_tif_to_png.py --verify
```

After conversion, the expected referenced PNG paths are:

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

The Maize Tassel Counting dataset can be downloaded from the [MTC GitHub page](https://github.com/poppinace/mtc). Save the downloaded archive as `Maize Tassel Counting Dataset.zip`, place it under `images/MTC/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    MTC/
      Maize Tassel Counting Dataset.zip
```

After extraction, the expected structure is:

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

The NuCLS dataset can be downloaded from [OpenDataLab](https://opendatalab.com/OpenDataLab/NuCLS/tree/main/raw). Save the downloaded archive as `NuCLS.tar.gz.00`, place it under `images/NuCLS/`, then extract it in place without flattening the extracted `NuCLS/` folder.

Before extraction:

```text
data/
  images/
    NuCLS/
      NuCLS.tar.gz.00
```

After extraction, the expected structure is:

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

The NuInsSeg dataset can be downloaded from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10518968). Save the downloaded archive as `NuInsSeg.zip`, place it under `images/NuInsSeg/`, then extract it in place without flattening the extracted organ folders.

Before extraction:

```text
data/
  images/
    NuInsSeg/
      NuInsSeg.zip
```

After extraction, the expected structure is:

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

The NWPU-CROWD dataset can be downloaded from the [NWPU-CROWD official website](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/). CLOC only needs the five image archives: `images_part1.zip` through `images_part5.zip`. Place these five archives under `images/NWPU-CROWD/`, then use the command below to extract them into one image folder.

Before extraction:

```text
data/
  images/
    NWPU-CROWD/
      images_part1.zip
      images_part2.zip
      ...
      images_part5.zip
```

Extract the five image-part archives into a single `NWPU-CROWD/` image folder:

```bash
cd data/images/NWPU-CROWD
mkdir -p NWPU-CROWD
for z in images_part*.zip; do
  unzip -q -o "$z" -d NWPU-CROWD
done
```

After extraction, the expected structure is:

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

The NWPU-MOC dataset can be downloaded from the [official GitHub page](https://github.com/lyongo/NWPU-MOC). Save the downloaded archive as `NWPU-MOC.zip`, place it under `images/NWPU-MOC/`, then extract it in place without flattening the extracted folder.

Before extraction:

```text
data/
  images/
    NWPU-MOC/
      NWPU-MOC.zip
```

After extraction, the expected structure is:

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

The NWPU-VHR-10 dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10). Place the downloaded archive under `images/NWPU-VHR-10/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    NWPU-VHR-10/
      archive.zip
```

After extraction, the expected structure is:

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

Objects365-2020 is downloaded from the official KS3 patch files. Note that the base URL is not a browsable web entry and may return 404 if opened directly in a browser; use the script below to download the concrete patch files. This dataset is large: the compressed download size of all patches is about 368 GB, and the extracted tree requires additional disk space.

Run the script below to download and extract all required patch files:

```bash
cd data
bash tools/downloaders/download_objects365_patches_no_proxy.sh
```

After extraction, the expected structure is:

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

Rebar Counting can be downloaded from [Roboflow](https://universe.roboflow.com/search?q=rebar%20counting). Search for the dataset named `rebar counting` by author `fyp2`, with about 250 images, then export/download the COCO-format archive. Place the downloaded archive under `images/rebar_counting/`, then extract it in place.

Before extraction:

```text
data/
  images/
    rebar_counting/
      rebar counting.coco.zip
```

After extraction, the expected structure is:

```text
rebar_counting/
  rebar counting.coco.zip
  train/
    *.jpg
    _annotations.coco.json
```

### RSOD

The RSOD dataset can be downloaded from [OpenDataLab](https://opendatalab.com/OpenDataLab/RSOD/tree/main/raw). Place the downloaded `RSOD.tar.gz` under `images/RSOD/`, then use the command below to extract the outer archive and the four class-specific inner zip files.

Before extraction:

```text
data/
  images/
    RSOD/
      RSOD.tar.gz
```

After extracting the outer archive:

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

Extract the four inner archives:

```bash
cd data/images/RSOD/RSOD
for z in aircraft.zip oiltank.zip overpass.zip playground.zip; do
  unzip -q -o "$z"
done
```

After extraction, the expected structure is:

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

The ShanghaiTech crowd counting dataset can be downloaded from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/xyyu18/shanghaitech-crowd-counting-dataset). Place the downloaded archive under `images/ShanghaiTech/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    ShanghaiTech/
      archive.zip
```

After extraction, the expected structure is:

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

The Soybean Pod Images from UAVs dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/jiajiali/uav-based-soybean-pod-images). Place the downloaded archive under `images/soybean_pod/`, then unzip it in place.

Before extraction:

```text
data/
  images/
    soybean_pod/
      archive.zip
```

After extraction, the expected raw structure is:

```text
data/
  images/
    soybean_pod/
      archive.zip
      dataset/
        *.bmp
        *.json
```

Soybean Pod Images from UAVs also needs PNG images generated from the BMP files. You can run the dataset-specific command below immediately, or run all conversion scripts together in the final "Conversion Stage":

```bash
python tools/converters/convert_soybean_pod_bmp_to_png.py --verify
```

After conversion, the expected structure is:

```text
soybean_pod/
  dataset_png/
    *.png
```

### UpCount

UpCount images can be downloaded from [Zenodo](https://zenodo.org/records/12683104/files/images.zip?download=1). Place the downloaded `images.zip` under `images/upcount/`, then extract it in place.

Before extraction:

```text
data/
  images/
    upcount/
      images.zip
```

After extraction, the expected structure is:

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

The VGG Cell Detection dataset can be downloaded from [Academic Torrents](https://academictorrents.com/details/b32305598175bb8e03c5f350e962d772a910641c). Place the downloaded archive under `images/VGG/`, then unzip it into `images/VGG/VGG_cells/`.

Before extraction:

```text
data/
  images/
    VGG/
      cells.zip
```

After extraction, the expected structure is:

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

The VisDrone dataset can be downloaded from the [Ultralytics YOLOv5 v1.0 release page](https://github.com/ultralytics/yolov5/releases/tag/v1.0). CLOC needs `VisDrone2019-DET-train.zip` and `VisDrone2019-DET-val.zip`. Place these two archives under `images/VisDrone/`, then extract them in place without flattening or renaming the extracted folders.

Before extraction:

```text
data/
  images/
    VisDrone/
      VisDrone2019-DET-train.zip
      VisDrone2019-DET-val.zip
```

After extraction, the expected structure begins as:

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

The VOC2007 dataset can be downloaded from the [PASCAL VOC2007 official page](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). Download `training/validation data` and `annotated test data`. Place both downloaded archives under `images/VOCdevkit/`, then extract them in place without flattening the extracted folder.

Before extraction:

```text
data/
  images/
    VOCdevkit/
      VOCtrainval_06-Nov-2007.tar
      VOCtest_06-Nov-2007.tar
```

After extraction, the expected structure is:

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

xView can be downloaded from [Kaggle](https://www.kaggle.com/datasets/hassanmojab/xview-dataset?resource=download). Place the downloaded archive under `images/xview/`, then extract it in place without flattening the extracted folders.

Before extraction:

```text
data/
  images/
    xview/
      archive.zip
```

After extraction, the expected structure is:

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

## Conversion Stage

This stage is required when preparing the complete CLOC dataset, unless you have already run all necessary per-dataset conversion commands from the dataset sections above. It converts non-standard source-image formats such as TIFF, BMP, NPY, or NIfTI into the PNG images referenced by the CLOC dataset JSON files.

The recommended workflow is to preview the conversion plan first, then run all conversions:

```bash
cd data
python tools/convert_all_sources.py
python tools/convert_all_sources.py --run
```

The first command is a dry-run. It checks whether each conversion step has the required inputs and writes `metadata/conversion_dry_run_report.json`. If the dry-run reports missing inputs, return to the corresponding dataset section and check the downloaded archive and extracted folder structure. The second command actually generates the converted images.

We also provide several optional commands:

- To overwrite existing converted outputs, run `python tools/convert_all_sources.py --run --overwrite`.
- To run only one dataset conversion, first list step ids with `python tools/convert_all_sources.py --list`, then pass `--only <step_id>`.
- The conversion plan is stored in `manifests/conversion_manifest.json`, and individual converter scripts are stored in `tools/converters/`. A per-dataset command shown above and the unified command in this stage are equivalent entry points, so they do not need to be run twice.

## Rebuild Stage

This stage is also required when preparing the complete CLOC dataset. CLOC annotations reference augmented images, including high-resolution tiles, cropped augmentation images, and stitched/mosaic augmentation images. Some of these augmented images are derived from source datasets with redistribution restrictions, so we do not directly distribute them. Instead, we provide reconstruction parameters and rebuild scripts so users can regenerate them locally from the downloaded source datasets.

After source dataset setup and conversion are complete, preview the rebuild plan first, then run the rebuild:

```bash
cd data
python tools/rebuild_restricted_derived_images.py --dry-run --verify
python tools/rebuild_restricted_derived_images.py --verify --overwrite
```

The first command does not write images; it checks reconstruction parameters, source images, and target paths. The second command writes rebuilt outputs under `augmented/` and does not modify the train/val/test JSON files. If the dry-run reports missing source images or reconstruction-parameter errors, fix the corresponding source dataset directory first.


## Final Audit Stage

The final audit is the last required stage. It should be run after source dataset extraction, conversion, and rebuild are complete. The audit only checks paths; it does not generate images or modify annotations. Its purpose is to verify that every `image_path` in the CLOC dataset train/val/test split JSON files resolves to an existing image inside the current `data/` workspace.

Run the audit:

```bash
cd data
python tools/audit_annotation_image_paths.py \
  --workspace . \
  --report metadata/annotation_image_path_audit_final.json \
  --markdown metadata/annotation_image_path_audit_final.md
```

The audit scans the unexpanded split files by default:

```text
annotations/train_split.json
annotations/val_split.json
annotations/test_split.json
```

The audit passes only when `missing` equals 0 for every split. If any split reports `missing > 0`, some annotation paths still cannot be resolved and the CLOC dataset is not fully prepared. In that case, first inspect `metadata/annotation_image_path_audit_final.md`, which summarizes missing paths by dataset root and includes sample missing records for debugging.
