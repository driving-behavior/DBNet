## DBNet-2018 Challenge
The DBNet-2018 challenge data are organized as follows:

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

Please note that the data in subfolders of `train/`, `val/` and `test/` are __continuous__ and __time-ordered__. The `ith` line of `behavior.csv` correponds to `i-1.jpg` in `dvf_66x200/` and `i-1.las` in `points_16384/`. Moreover, if you don't intend to utilize the prepared data directly, please download and pre-process the raw data in your favorite methods.
