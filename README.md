# TorchProf_XDU

因[课程作业](https://github.com/Livinfly/torchprof-XDU)而诞生，但因为原版 [TorchProf](https://github.com/awwong1/torchprof) 比较老，为了更好的实验环境，用AI工具对其使用的profiler后端 PyTorch autograd profiler，更换为 Pytorch profiler，本人测试的情况下，和原版使用一致，欢迎人工对比两个仓库的功能实现并补充。

- 同时，添加了 Flops 和 Params 的测定，按照某种标准排序，不过存在不保留前缀的情况，存在问题（待修复），建议对照原版输出和模型结构去寻找对应层。
- 现又增加新功能，为了更好的后续的作图等工作，新实现 `get_raw_measure_dict_from_profiler_data`，支持取出 `dict{layer, measure}`，作为后续处理的基础，使用方法见 `torchprof_xdu_profile_detailed_raw_example.py`，可能需要对导出的单位进行后处理。
- 现在对 cuda 模式进行修复，`_build_measure_tuple`。
  优先使用 detailed 版本
- 现由 [Xorzj](https://github.com/xorzj) 更新画图（柱状图和饼状图）功能，具体调用方法可见课程作业。

```bash
# 在torchprof_xdu文件外，使用下面这行命令来测试
# 注意，同文件夹下不能出现同名 profile 文件/文件夹！！
python -m torchprof_xdu.examples.torchprof_xdu_profile_example
python -m torchprof_xdu.examples.torchprof_xdu_profile_detailed_example
python -m torchprof_xdu.examples.torchprof_xdu_profile_detailed_raw_example
```

```python
# 临时调用
current_file_path = os.path.abspath(__file__)
task_dir = os.path.dirname(current_file_path)
profile_dir = os.path.dirname(task_dir)
project_root = os.path.dirname(profile_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torchprof_xdu import ProfileDetailed
```

```bash
# profile
Module         | Self CPU total | CPU total | Self CPU Mem | CPU Mem   | Number of Calls
---------------|----------------|-----------|--------------|-----------|----------------
AlexNet        |                |           |              |           |
├── features   |                |           |              |           |
│├── 0         | 67.121us       | 267.032us | 22.16 Mb     | 110.78 Mb | 30
│├── 1         | 1.355us        | 2.311us   |              |           | 30
│├── 2         | 25.063us       | 49.833us  | 5.34 Mb      | 21.36 Mb  | 30
│├── 3         | 57.718us       | 229.674us | 16.02 Mb     | 80.09 Mb  | 30
│├── 4         | 1.191us        | 1.989us   |              |           | 30
│├── 5         | 18.598us       | 36.910us  | 3.71 Mb      | 14.85 Mb  | 30
│├── 6         | 50.601us       | 201.087us | 7.43 Mb      | 37.13 Mb  | 30
│├── 7         | 1.117us        | 1.837us   |              |           | 30
│├── 8         | 50.392us       | 200.273us | 4.95 Mb      | 24.76 Mb  | 30
│├── 9         | 1.105us        | 1.792us   |              |           | 30
│├── 10        | 33.257us       | 131.722us | 4.95 Mb      | 24.76 Mb  | 30
│├── 11        | 1.060us        | 1.716us   |              |           | 30
│└── 12        | 7.275us        | 14.254us  | 1.05 Mb      | 4.22 Mb   | 30
├── avgpool    | 2.338us        | 4.934us   | 1.05 Mb      | 3.16 Mb   | 30
└── classifier |                |           |              |           |
 ├── 0         | 8.559us        | 18.119us  | 1.05 Mb      | 4.22 Mb   | 30
 ├── 1         | 208.957us      | 418.467us | 480.00 Kb    | 960.00 Kb | 30
 ├── 2         | 950.800ns      | 1.384us   |              |           | 30
 ├── 3         | 6.185us        | 13.400us  | 480.00 Kb    | 1.88 Mb   | 30
 ├── 4         | 97.181us       | 195.376us | 480.00 Kb    | 960.00 Kb | 30
 ├── 5         | 761.704ns      | 1.128us   |              |           | 30
 └── 6         | 25.808us       | 52.271us  | 117.19 Kb    | 234.38 Kb | 30

# profile_detailed
Module            | Self CPU total | CPU total | Self CPU Mem | CPU Mem   | FLOPs        | Parameters | Calls
------------------|----------------|-----------|--------------|-----------|--------------|------------|------
AlexNet           |                |           |              |           |              |            |
   ├── features   |                |           |              |           |              |            |
   │  ├── 0       | 62.856us       | 249.814us | 22.16 Mb     | 110.78 Mb | 4.22GFLOPs   | 23.30K     | 30
   │  ├── 1       | 1.411us        | 2.368us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
   │  ├── 2       | 23.392us       | 46.480us  | 5.34 Mb      | 21.36 Mb  | 0 FLOPs      | 0          | 30
   │  ├── 3       | 49.717us       | 197.516us | 16.02 Mb     | 80.09 Mb  | 13.44GFLOPs  | 307.39K    | 30
   │  ├── 4       | 1.291us        | 2.159us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
   │  ├── 5       | 14.808us       | 29.333us  | 3.71 Mb      | 14.85 Mb  | 0 FLOPs      | 0          | 30
   │  ├── 6       | 45.768us       | 181.742us | 7.43 Mb      | 37.13 Mb  | 6.73GFLOPs   | 663.94K    | 30
   │  ├── 7       | 1.246us        | 2.030us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
   │  ├── 8       | 46.505us       | 184.646us | 4.95 Mb      | 24.76 Mb  | 8.97GFLOPs   | 884.99K    | 30
   │  ├── 9       | 1.153us        | 1.881us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
   │  ├── 10      | 31.768us       | 125.760us | 4.95 Mb      | 24.76 Mb  | 5.98GFLOPs   | 590.08K    | 30
   │  ├── 11      | 1.080us        | 1.766us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
   │  └── 12      | 7.603us        | 14.902us  | 1.05 Mb      | 4.22 Mb   | 0 FLOPs      | 0          | 30
   ├── avgpool    | 2.669us        | 5.562us   | 1.05 Mb      | 3.16 Mb   | 0 FLOPs      | 0          | 30
   └── classifier |                |           |              |           |              |            |
      ├── 0       | 65.200ns       | 65.200ns  | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
      ├── 1       | 204.598us      | 409.705us | 480.00 Kb    | 960.00 Kb | 2.26GFLOPs   | 37.75M     | 30
      ├── 2       | 933.900ns      | 1.406us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
      ├── 3       | 63.500ns       | 63.500ns  | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
      ├── 4       | 96.719us       | 194.083us | 480.00 Kb    | 960.00 Kb | 1.01GFLOPs   | 16.78M     | 30
      ├── 5       | 933.300ns      | 1.370us   | 0 b          | 0 b       | 0 FLOPs      | 0          | 30
      └── 6       | 25.973us       | 52.549us  | 117.19 Kb    | 234.38 Kb | 245.76MFLOPs | 4.10M      | 30

# profile_detailed (show_events=True)
Module                                     | Self CPU total | CPU total | Self CPU Mem | CPU Mem   | FLOPs        | Calls
-------------------------------------------|----------------|-----------|--------------|-----------|--------------|------
AlexNet                                    |                |           |              |           |              |
   ├── features                            |                |           |              |           |              |
   │  ├── 0                                |                |           |              |           |              |
   │  │  ├── aten::conv2d                  | 286.800ns      | 50.508us  | 0 b          | 22.16 Mb  | 4.22GFLOPs   | 30
   │  │  ├── aten::convolution             | 359.700ns      | 50.221us  | 0 b          | 22.16 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::_convolution            | 432.900ns      | 49.862us  | 0 b          | 22.16 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::mkldnn_convolution      | 48.882us       | 49.429us  | 0 b          | 22.16 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::empty                   | 296.000ns      | 296.000ns | 22.16 Mb     | 22.16 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::as_strided_             | 203.000ns      | 203.000ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::resize_                 | 47.800ns       | 47.800ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 1                                |                |           |              |           |              |
   │  │  ├── aten::relu_                   | 401.400ns      | 1.276us   | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::clamp_min_              | 874.500ns      | 874.500ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 2                                |                |           |              |           |              |
   │  │  ├── aten::max_pool2d              | 296.100ns      | 22.551us  | -10.68 Mb    | 5.34 Mb   | 0 FLOPs      | 30
   │  │  └── aten::max_pool2d_with_indices | 22.255us       | 22.255us  | 16.02 Mb     | 16.02 Mb  | 0 FLOPs      | 30
   │  ├── 3                                |                |           |              |           |              |
   │  │  ├── aten::conv2d                  | 246.300ns      | 44.590us  | 0 b          | 16.02 Mb  | 13.44GFLOPs  | 30
   │  │  ├── aten::convolution             | 272.700ns      | 44.344us  | 0 b          | 16.02 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::_convolution            | 332.700ns      | 44.071us  | 0 b          | 16.02 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::mkldnn_convolution      | 43.347us       | 43.738us  | 0 b          | 16.02 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::empty                   | 217.700ns      | 217.700ns | 16.02 Mb     | 16.02 Mb  | 0 FLOPs      | 30
   │  │  ├── aten::as_strided_             | 138.700ns      | 138.700ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::resize_                 | 34.300ns       | 34.300ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 4                                |                |           |              |           |              |
   │  │  ├── aten::relu_                   | 364.900ns      | 1.016us   | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::clamp_min_              | 651.500ns      | 651.500ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 5                                |                |           |              |           |              |
   │  │  ├── aten::max_pool2d              | 264.099ns      | 13.661us  | -7.43 Mb     | 3.71 Mb   | 0 FLOPs      | 30
   │  │  └── aten::max_pool2d_with_indices | 13.397us       | 13.397us  | 11.14 Mb     | 11.14 Mb  | 0 FLOPs      | 30
   │  ├── 6                                |                |           |              |           |              |
   │  │  ├── aten::conv2d                  | 251.900ns      | 35.781us  | 0 b          | 7.43 Mb   | 6.73GFLOPs   | 30
   │  │  ├── aten::convolution             | 275.001ns      | 35.529us  | 0 b          | 7.43 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::_convolution            | 355.300ns      | 35.254us  | 0 b          | 7.43 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::mkldnn_convolution      | 34.478us       | 34.899us  | 0 b          | 7.43 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::empty                   | 239.600ns      | 239.600ns | 7.43 Mb      | 7.43 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::as_strided_             | 145.500ns      | 145.500ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::resize_                 | 35.400ns       | 35.400ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 7                                |                |           |              |           |              |
   │  │  ├── aten::relu_                   | 378.200ns      | 1.009us   | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::clamp_min_              | 631.000ns      | 631.000ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 8                                |                |           |              |           |              |
   │  │  ├── aten::conv2d                  | 233.301ns      | 40.513us  | 0 b          | 4.95 Mb   | 8.97GFLOPs   | 30
   │  │  ├── aten::convolution             | 261.000ns      | 40.280us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::_convolution            | 328.400ns      | 40.019us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::mkldnn_convolution      | 39.324us       | 39.690us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::empty                   | 210.500ns      | 210.500ns | 4.95 Mb      | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::as_strided_             | 125.900ns      | 125.900ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::resize_                 | 30.400ns       | 30.400ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 9                                |                |           |              |           |              |
   │  │  ├── aten::relu_                   | 352.799ns      | 910.200ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::clamp_min_              | 557.401ns      | 557.401ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 10                               |                |           |              |           |              |
   │  │  ├── aten::conv2d                  | 235.700ns      | 29.279us  | 0 b          | 4.95 Mb   | 5.98GFLOPs   | 30
   │  │  ├── aten::convolution             | 252.100ns      | 29.043us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::_convolution            | 316.300ns      | 28.791us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::mkldnn_convolution      | 28.127us       | 28.475us  | 0 b          | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::empty                   | 200.700ns      | 200.700ns | 4.95 Mb      | 4.95 Mb   | 0 FLOPs      | 30
   │  │  ├── aten::as_strided_             | 119.300ns      | 119.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::resize_                 | 28.200ns       | 28.200ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  ├── 11                               |                |           |              |           |              |
   │  │  ├── aten::relu_                   | 339.099ns      | 854.297ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  │  └── aten::clamp_min_              | 515.198ns      | 515.198ns | 0 b          | 0 b       | 0 FLOPs      | 30
   │  └── 12                               |                |           |              |           |              |
   │  │  ├── aten::max_pool2d              | 281.901ns      | 7.219us   | -2.11 Mb     | 1.05 Mb   | 0 FLOPs      | 30
   │  │  └── aten::max_pool2d_with_indices | 6.937us        | 6.937us   | 3.16 Mb      | 3.16 Mb   | 0 FLOPs      | 30
   ├── avgpool                             |                |           |              |           |              |
   │  ├── aten::adaptive_avg_pool2d        | 222.801ns      | 2.609us   | 0 b          | 1.05 Mb   | 0 FLOPs      | 30
   │  ├── aten::_adaptive_avg_pool2d       | 1.925us        | 2.386us   | 0 b          | 1.05 Mb   | 0 FLOPs      | 30
   │  ├── aten::empty                      | 96.500ns       | 96.500ns  | 0 b          | 0 b       | 0 FLOPs      | 30
   │  └── aten::resize_                    | 364.701ns      | 364.701ns | 1.05 Mb      | 1.05 Mb   | 0 FLOPs      | 30
   └── classifier                          |                |           |              |           |              |
      ├── 0                                |                |           |              |           |              |
      │  └── aten::dropout                 | 61.600ns       | 61.600ns  | 0 b          | 0 b       | 0 FLOPs      | 30
      ├── 1                                |                |           |              |           |              |
      │  ├── aten::linear                  | 526.499ns      | 199.139us | 0 b          | 480.00 Kb | 0 FLOPs      | 30
      │  ├── aten::t                       | 437.500ns      | 795.100ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::transpose               | 252.900ns      | 357.600ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::as_strided              | 125.300ns      | 125.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::addmm                   | 197.303us      | 197.818us | 480.00 Kb    | 480.00 Kb | 2.26GFLOPs   | 30
      │  ├── aten::expand                  | 127.800ns      | 148.400ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::copy_                   | 342.200ns      | 342.200ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  └── aten::resolve_conj            | 23.900ns       | 23.900ns  | 0 b          | 0 b       | 0 FLOPs      | 30
      ├── 2                                |                |           |              |           |              |
      │  ├── aten::relu_                   | 395.901ns      | 772.203ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  └── aten::clamp_min_              | 376.302ns      | 376.302ns | 0 b          | 0 b       | 0 FLOPs      | 30
      ├── 3                                |                |           |              |           |              |
      │  └── aten::dropout                 | 77.500ns       | 77.500ns  | 0 b          | 0 b       | 0 FLOPs      | 30
      ├── 4                                |                |           |              |           |              |
      │  ├── aten::linear                  | 518.000ns      | 91.939us  | 0 b          | 480.00 Kb | 0 FLOPs      | 30
      │  ├── aten::t                       | 418.600ns      | 799.900ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::transpose               | 262.900ns      | 381.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::as_strided              | 139.000ns      | 139.000ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::addmm                   | 90.172us       | 90.621us  | 480.00 Kb    | 480.00 Kb | 1.01GFLOPs   | 30
      │  ├── aten::expand                  | 121.500ns      | 142.100ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  ├── aten::copy_                   | 290.000ns      | 290.000ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  └── aten::resolve_conj            | 17.100ns       | 17.100ns  | 0 b          | 0 b       | 0 FLOPs      | 30
      ├── 5                                |                |           |              |           |              |
      │  ├── aten::relu_                   | 468.400ns      | 809.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
      │  └── aten::clamp_min_              | 340.900ns      | 340.900ns | 0 b          | 0 b       | 0 FLOPs      | 30
      └── 6                                |                |           |              |           |              |
         ├── aten::linear                  | 400.902ns      | 23.882us  | 0 b          | 117.19 Kb | 0 FLOPs      | 30
         ├── aten::t                       | 407.000ns      | 776.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
         ├── aten::transpose               | 254.900ns      | 369.300ns | 0 b          | 0 b       | 0 FLOPs      | 30
         ├── aten::as_strided              | 134.000ns      | 134.000ns | 0 b          | 0 b       | 0 FLOPs      | 30
         ├── aten::addmm                   | 22.295us       | 22.704us  | 117.19 Kb    | 117.19 Kb | 245.76MFLOPs | 30
         ├── aten::expand                  | 121.300ns      | 140.900ns | 0 b          | 0 b       | 0 FLOPs      | 30
         ├── aten::copy_                   | 250.201ns      | 250.201ns | 0 b          | 0 b       | 0 FLOPs      | 30
         └── aten::resolve_conj            | 17.900ns       | 17.900ns  | 0 b          | 0 b       | 0 FLOPs      | 30
```
