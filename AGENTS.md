# 项目协作说明

## 回答风格

在回答时，尽量避免使用“引号+A-B-C”的表达方式。

除非必要，减少冒号、破折号、分号。

避免使用超长复合句。

避免在非数据公式表述中使用奇怪的符号，例如复杂箭头和不常用数学符号。

减少使用“链路”“回路”“闭环”“耦合”等复杂、生涩词汇。

所有句子的过渡词和连接词都使用最基础、常用的词语。

在进行总结、框架等任务时，可以适当使用分列表、小标题来使信息组织更有条理。

## 实验命名规则

新增实验要尽量参考现有 `图像实验` 和 `视频实验` 的文件组织方式。

不要随意新增过度工程化的目录名。

结果统一保存到当前实验文件夹下的 `results`。

如果用户没有明确要求，不要新增 `figures`、`methods`、`outputs` 这类新目录。

脚本命名要短，并且和已有方法名一致。

方法目录命名要保持一致，例如：

- `GLSKF`
- `GLSKF-T`
- `GLTC-TNN`
- `HaLRTC`
- `SPC`

## 随机低秩张量实验命名规则

第三阶段实验文件夹命名为：

`随机低秩张量实验`

目录结构固定为：

```text
随机低秩张量实验/
  data/
  results/
  lowrank.py
  GLSKF.ipynb
  GLTC-TNN.ipynb
  HaLRTC.ipynb
  SPC.m
  GLSKF-T.m
  SPC/
  GLSKF-T/
```

其中 `lowrank.py` 只负责生成随机低秩张量和掩码。

`summarize.py` 只有在用户明确要求汇总和画图时再创建。

不要提前创建 `figures` 文件夹。

不要提前创建 `methods` 文件夹。

## 随机低秩张量实验数据命名

生成的完整张量保存为：

```text
S920.mat
S921.mat
S922.mat
```

带缺失掩码的数据保存为：

```text
S920_miss80.mat
S920_miss90.mat
S920_miss95.mat
```

Python 需要读取的数据可以同时保存同名 `.npz`。

MATLAB 和 Python 读取的数据命名要保持一致。

每个数据文件至少保存：

- `X`
- `Omega`
- `Y`
- `seed`
- `missing_rate`
- `tubal_rank`
- `tensor_size`

完整张量文件还要保存：

- `U`
- `S`
- `V`

## 随机低秩张量实验结果命名

结果保存到：

```text
随机低秩张量实验/results/
```

每个方法一个子目录：

```text
results/GLSKF/
results/GLSKF-T/
results/GLTC-TNN/
results/HaLRTC/
results/SPC/
```

每个实验组合的目录命名为：

```text
S920_miss80
S920_miss90
S920_miss95
```

每个组合至少保存：

- `最佳迭代.csv`
- `metrics.txt`
- `result.mat` 或 `result.npz`
- `实验总结.xlsx`

`最佳跌打.csv` 的字段尽量统一为：

- `dataset`
- `method`
- `variant`
- `seed`
- `missing_rate`
- `iteration`
- `elapsed_time_seconds`
- `MSE`
- `RMSE`
- `RSE`
- `MAE`
- `relative_change`
- `parameter_settings`
- `convergence_status`

随机低秩张量实验不以 SSIM 作为主要指标。

不计算PSNR

主要指标使用：

- `MSE`
- `RMSE`
- `RSE`
- `MAE`
- `time`
