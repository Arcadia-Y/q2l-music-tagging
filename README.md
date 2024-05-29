# Query2Label Music Tagging

This is the repo for the paper [**Improving CNN-based Music Tagging with Query2Label Transformer**](./paper.pdf).

We have a pretrained q2l model at ``./pretrained/q2l.pth``. You can use it to [predict](#prediction) tags for your own music.

To train or evaluate models, you should first download the dataset [MagnaTagAtune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) (only mp3 files are needed), put it in ``YOUR_DATA_PATH/mp3/``, and [preprocess](#preprocessing) it.

## Possible Issues
- **Using torch under 2.0 and cuda under 11.8 may result in CUFFT_INTERNAL_ERROR, especially on RTX 4090.**

- If you don't have ffmpeg for processing mp3, ``apt install ffmpeg`` first.

## Models
- **Musicnn** : End-to-end Learning for Music Audio Tagging at Scale, Pons et al., 2018 [[arxiv](https://arxiv.org/abs/1711.02520)]
- **Harmonic CNN** : Data-Driven Harmonic Filters for Audio Representation Learning, Won et al., 2020 [[pdf](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf)]
- **Short-chunk CNN + Residual** : Short-chunk CNN with residual connections, i.e. "ResCNN" in our paper.
- **Query2Label**: Our model which combines modified ResCNN and Q2L transformer.


## Requirements
```
conda create -n YOUR_ENV_NAME python=3.9
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Preprocessing
STFT will be done on-the-fly. You only need to read and resample audio files into `.npy` files. 

`cd preprocessing/`

`python -u mtat_read.py run YOUR_DATA_PATH`

## Training

`cd training/`

`python -u main.py --data_path YOUR_DATA_PATH`

Options

```
'--num_workers', type=int, default=0
'--model_type', type=str, default='q2l',
				choices=['musicnn', 'short_res', 'hcnn', 'q2l']
'--n_epochs', type=int, default=15
'--batch_size', type=int, default=32
'--lr', type=float, default=1e-4
'--use_tensorboard', type=int, default=1
'--model_save_path', type=str, default='./models'
'--model_load_path', type=str, default='.'
'--data_path', type=str, default='./data'
'--log_step', type=int, default=20
```

## Evaluation
`cd training/`

`python -u eval.py --data_path YOUR_DATA_PATH`

Options

```
'--num_workers', type=int, default=0
'--model_type', type=str, default='q2l',
				choices=['musicnn', 'short_res', 'hcnn', 'q2l']
'--batch_size', type=int, default=32
'--model_load_path', type=str, default='.'
'--data_path', type=str, default='./data'
```

## Prediction
`python -u predict.py --input INPUT_AUDIO_PATH`

Options
```
'--output_format', type=str, default='image', choices=['image', 'JSON']
'--model_load_path', type=str, default='./pretrained/q2l.pth'
```

## References
Most codes are based on the following two work:

**Evaluation of CNN-based Automatic Music Tagging Models**, SMC 2020 [[arxiv](https://arxiv.org/abs/2006.00751)], [github repo](https://github.com/minzwon/sota-music-tagging-models/)

**Query2Label: A Simple Transformer Way to Multi-Label Classification**, [[arxiv](https://arxiv.org/abs/2107.10834)], [github repo](https://github.com/SlongLiu/query2labels/)
