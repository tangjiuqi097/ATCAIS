

# ATCAIS

It is the ATCAIS tracker for VOT-RGBD2019 challenge.


The code is implemented based on 
[ATOM](https://github.com/visionml/pytracking) 
and [HTC](https://github.com/open-mmlab/mmdetection).

## results

The F1-score is 0.7016 while the AO is 0.6463. 

The results are available at [./vot_rgbd2019_result](https://github.com/tangjiuqi097/ATCAIS/tree/master/vot_rgbd2019_result).

# usage

run tracker_ATCAIS.m by VOT-toolkit, and you should adjust the path to your path.

## installation

### dependencies

system Ubuntu16.04

GPU 1080ti

[cuda9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal)

[cudnn7.0.5](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse705-9)

[anaconda3.7](https://www.anaconda.com/distribution/#download-section)


### setup python environments

    cd /path/to/this_code/
    sh install_for_vot.sh
if it raise errors, you can run the command in it one by one


### download models


download the atom checkpoint and move it to 
'pytracking/networks'

[Baidu Netdisk](https://pan.baidu.com/s/1zbrtuX6X2rRJ5eCEFj-OCA) 
password: mwcp


download the detection checkpoint and move it to 
'pytracking/mmdetection/checkpoints'

[Baidu Netdisk](https://pan.baidu.com/s/1YYJ0I4UECRnYf95A2nHUSw)
password:  njw1 


### setup path

change the `config_file` and `checkpoint_file` to corresponding absolute path in the file 
`pytracking/tracker/ATCAIS/det_mmdet.py`

change the path in the file `tracker_ATCAIS.m` to corresponding absolute path, and move it to the vot workspace

change the `settings.results_path` and `settings.network_path` to corresponding absolute path in the file `pytracking/evaluation/local`


### Troublesome

For the first time to run, it will build the prroi_pooling and may be quit.
It may be timeout.
You can simply just try again.


if you meet the following error:
Tracker execution interrupted: Invalid MEX-file '/home/tangjiuqi097/data/vot2019/vot-toolkit/native/traxclient.mexa64': 
Missing symbol '_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc' required by '/home/tangjiuqi097/data/vot2019/vot-toolkit/native/traxclient.mexa64'
Missing symbol '_ZNSt13runtime_errorC1EPKc' required by '/home/tangjiuqi097/data/vot2019/vot-toolkit/native/traxclient.mexa64'
Missing symbol '_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm' required by '/home/tangjiuqi097/data/vot2019/vot-toolkit/native/traxclient.mexa64'
Missing symbol '_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_' required by '/home/tangjiuqi097/data/vot2019/vot-toolkit/native/traxclient.mexa64'
...

I am not sure what happens, but you can run 

`export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libprotobuf.so.9` 

before you open matlab to solve it.


if you have any other questions, please contact to me
`wym097@mail.dlut.edu.cn`.
if you have not got reply in time, try
`1714079799@qq.com`





