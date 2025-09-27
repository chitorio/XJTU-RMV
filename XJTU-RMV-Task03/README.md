# 第三次培训任务

## 遇到的问题
1. 再次出现了clangd搜索不到opencv头文件的错误。通过修改vscode settings.json文件添加opencv库位置解决，不过在添加完之后出现了vscode的“无法在合理时间内解析shell环境”报错，是因为添加了clangd直接利用shell搜索opencv头文件位置导致的，不过貌似没有影响。ceres库没有受到clangd影响，或许是安装路径问题。
2. 物理坐标轴和视频坐标轴不一致（体现在y轴方向上）导致ceres优化出现较大问题，将初速度向上的扳回向下。