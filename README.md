# TorchProf_XDU

因为原版 [TorchProf](https://github.com/awwong1/torchprof) 比较老，为了更好的实验环境，用AI工具对其使用的profiler后端 PyTorch autograd profiler，更换为 Pytorch profiler，本人测试的情况下，和原版使用一致，欢迎人工对比两个仓库的功能实现并补充。

```bash
# 在torchprof_xdu文件外，使用下面这行命令来测试
python -m torchprof_xdu.examples.torchprof_xdu_profile_example
```
