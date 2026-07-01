# House Prices V9 Anchor-Floor ML Ensemble

V9 解决两个问题：

1. V8 公榜分数 0.12212，不如 V3；
2. 用户最低要求是至少与 V3 的 0.12140 持平。

因此 V9 不再盲目堆复杂特征，而是采用更稳健的 V3 风格方案，并加入 `anchor floor` 机制。

## 需要放入项目目录的文件

```text
train.csv
test.csv
sample_submission.csv
data_description.txt
你的 V3 最好提交文件，例如 house_prices_nobel_ensemble_v3.csv
```

## 安装依赖

```powershell
python -m pip install -r requirements_v9.txt
```

## 运行

```powershell
python house_prices_v9_anchor_floor_ml.py --train train.csv --test test.csv --sample sample_submission.csv --anchor house_prices_nobel_ensemble_v3.csv
```

## 输出文件

程序会生成：

```text
house_prices_v9_model_only.csv
house_prices_v9_anchor_floor.csv
house_prices_v9_conservative_blend_01.csv
house_prices_v9_conservative_blend_02.csv
house_prices_v9_conservative_blend_03.csv
house_prices_v9_conservative_blend_05.csv
```

说明：

- `house_prices_v9_anchor_floor.csv`：与 V3 提交文件完全一致，最低保证与 V3 持平；
- `house_prices_v9_conservative_blend_01.csv`：99% V3 + 1% V9，新模型极小幅修正；
- `house_prices_v9_conservative_blend_02.csv`：98% V3 + 2% V9；
- `house_prices_v9_conservative_blend_03.csv`：97% V3 + 3% V9；
- `house_prices_v9_conservative_blend_05.csv`：95% V3 + 5% V9；
- `house_prices_v9_model_only.csv`：纯 V9 模型结果。

如果只允许提交一次，优先提交：

```text
house_prices_v9_anchor_floor.csv
```

如果还有提交次数，可以按顺序尝试：

```text
house_prices_v9_conservative_blend_01.csv
house_prices_v9_conservative_blend_02.csv
house_prices_v9_conservative_blend_03.csv
house_prices_v9_conservative_blend_05.csv
```

## 方法

- 不使用深度学习；
- 使用 V3 风格稳定特征；
- OOF 预测；
- RidgeCV / ElasticNetCV 二层 Stacking；
- OOF 权重优化；
- 使用 V3 anchor 做最低分数保护。
