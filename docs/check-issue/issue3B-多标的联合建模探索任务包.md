# issue3B：多标的联合建模探索任务包

## 一、任务定位

`issue3B` 负责沿多标的联合建模路线做探索。

这条路线要回答的问题是：

1. 多个 ticker 一起训练是否比单股更有价值？
2. 联合建模时，是否需要把 `symbol / group` 加入上下文特征？
3. pooled 路线到底学到了共性信号，还是只是学到了市场组别差异？

## 二、任务目标

目标不是“把所有股票混起来训练一次”，而是明确 pooled 路线有没有保留价值。

建议至少比较两个设置：

1. 仅数值特征
2. 数值特征 + `symbol/group` 上下文特征

## 三、建议模型范围

建议至少比较：

1. `Logistic Regression`
2. `XGBoost`

## 四、建议实验设计

建议最小实验矩阵：

1. pooled + numeric only + Logistic
2. pooled + numeric only + XGBoost
3. pooled + context features + Logistic
4. pooled + context features + XGBoost

## 五、输出要求

至少输出：

1. pooled 实验代码
2. 场景级指标表
3. 按 ticker 拆解的指标表
4. 一页 pooled 路线结论

## 六、必须回答的问题

1. pooled 是否比单股更稳定？
2. pooled 加入 `symbol/group` 是否有帮助？
3. pooled 结果是否在所有 ticker 上都成立？
4. pooled 路线是否值得进入最终模型选择？

## 七、完成标准

完成标准不是“训练了联合模型”，而是：

1. pooled 结果可复现
2. 能区分 numeric only 与 context 方案
3. 能按 ticker 拆开解释结果
4. 能与 `issue3A` 做可比对结论

## 八、协作接口

需要交给 `issue4` 的内容：

1. pooled 场景结果表
2. per-symbol 结果表
3. 是否建议保留 context features
4. pooled 路线是否建议进入最终候选

## 九、风险提醒

1. 不要只看 pooled 总体 accuracy，要按 ticker 拆解。
2. 不要忽视中美市场交易结构差异。
3. 不要因为 pooled 总体分数略高，就默认它更好。
4. 不要把 context 特征当成“自动提高模型”的手段，必须解释它带来了什么。
