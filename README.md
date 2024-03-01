# Surface-VQMAE 
## Install
### Environment
```bash
conda env create -f env.yaml -n surf
conda activate surf
```
The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Datasets and Trained Weights
Data version: **2023.09.23**

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). 
Extract `all_structures.zip` into the `data` folder.  The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). 
You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/). 
Trained model weights are available [**here** (Hugging Face)](https://huggingface.co/luost26/DiffAb/tree/main) or [**here** (Google Drive)](https://drive.google.com/drive/folders/15ANqouWRTG2UmQS_p0ErSsrKsU4HmNQc?usp=sharing).


## Data Preprocessing
```markdown
python preprocess.py
```

## Pretraining
```bash
python train.py ./configs/train/vae.yml
```

## Train
```bash
python train.py ./configs/train/surfformer.yml
```

[//]: # (## Reference)

[//]: # ()
[//]: # (```bibtex)

[//]: # (```)
