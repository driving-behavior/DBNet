## Tools to process point clouds and images

A list of tools (python scripts) are created to make data processing easier and more convenient. Note that they are not professional tools so you may need to modify some lines before using it in your cases.

If you have more efficient tools, code or other suggestions to process DBNet data, especially point clouds, don't hesitate to contact [@wangjksjtu(wangjksjtu@gmail.com)](https://github.com/wangjksjtu) or __submit pull-requests directly__. 
Your contributions are highly encouraged and appreciated!

- __img_pre.py__: croping and resizing images using python-opencv
- __las2fmap.py__: extracting feature maps from point clouds
- __pcd2las.py__: downsampling point clouds; converting point clouds from '.pcd' to '.las' format. 
- __video2img.py__: converting one video to continuous frames

To see HELP for these script:

    python train.py -h

### Requirements
- python-opencv
- numpy, pickle, scipy, __laspy__
- __CloudCompare (CC)__ (set __PATH variables__)

### CC Examples
Convert point clouds to `.las` format:

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT LAS -O %s
    
Downsample point clouds to 16384 points and save in `.las` format:

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT LAS -O %s -SS RANDOM 16384

More command line usages of CloudCompareare available on the [offical manual page](http://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode).
