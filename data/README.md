## Home Directory of DBNet data

This is the place where DBNet data are placed in order to fit the default path in `../provider.py`. In total, two kinds of prepared data are provided, which are listed in `dbnet-2018` and `demo` folder, respectively.

### dbnet-2018
Download DBNet-2018 challenge data [here]() and organize the folders as follows (in `dbnet-2018/`):
```
├── train
├─  └── i [56 folders] (6569 in total, will release continously)
├─      ├── dvr_66x200   [<= 120 images]
├─      ├── points_16384 [<= 120 images]
├─      └── behavior.csv [labels]
├── val
├─  └── j [20 folders] (2349 in total)
├─      ├── dvr_66x200   [<= 120 images]
├─      ├── points_16384 [<= 120 clouds]
├─      └── behavior.csv [labels]
└── test
    └── k [20 folders] (2376 in total)
        ├── dvr_66x200   [<= 120 images]
        └── points_16384 [<= 120 clouds]
    
```
In general, the train/val/test ratio is approximatingly set to 8:1:1 and all of the val/test data are released already. Almost five eighths of training data are still pre-processed and will be __uploaded soon__.

Please note that the data in subfolders of `train/`, `val/` and `test/` are __continuous__ and __time-ordered__. The `ith` line of `behavior.csv` correponds to `i-1.jpg` in `dvf_66x200/` and `i-1.las` in `points_16384/`. Moreover, if you don't intend to utilize prepared data directly, please download and pre-process the [raw data]() in your favorite methods.

### demo
Download DBNet demo data [here]() and organize the folders as follows (in `demo`):

```
├── data.csv
├── DVR
├─  └── i.jpg [3788 images]
├── fmap
├─  └── i.jpg [3788 feature maps]
└── points_16384
    └── i.las [3788 point clouds]
```
