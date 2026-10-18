# 原easy 100→500开发失败案例核对（只读）

依据已有固定100图开发轨迹，无新环境执行或梯度更新。不剔除退步地图，不按开发成绩改变训练图选择。

## fifo

成功数53→49；从成功变失败6图，从失败变成功2图。失败停止原因：{'decision_limit': 51}。联合覆盖≥99%但任务未成功：4图。

|成功转失败地图|500图端点停止原因|决策数|联合覆盖|较弱本地覆盖|
|---|---|---:|---:|---:|
|DungeonMaps/train/easy/img_9642.png|decision_limit|196|1.0000|0.8998|
|DungeonMaps/train/easy/img_7312.png|decision_limit|196|1.0000|0.9821|
|DungeonMaps/train/easy/img_4424.png|decision_limit|196|1.0000|0.8516|
|DungeonMaps/train/easy/img_2352.png|decision_limit|196|0.8638|0.8638|
|DungeonMaps/train/easy/img_1964.png|decision_limit|196|1.0000|0.8463|
|DungeonMaps/train/easy/img_1566.png|decision_limit|196|0.7688|0.6702|

## ab

成功数55→57；从成功变失败1图，从失败变成功3图。失败停止原因：{'decision_limit': 43}。联合覆盖≥99%但任务未成功：9图。

|成功转失败地图|500图端点停止原因|决策数|联合覆盖|较弱本地覆盖|
|---|---|---:|---:|---:|
|DungeonMaps/train/easy/img_6936.png|decision_limit|196|1.0000|0.9683|

500图工程预算、继承、冻结、几何/通信检查已通过，未见需要先改算法或重置基线的证据。以上为终点现象分类，不能确定失败因果。下一轮保留同一规则、相同预算和全部开发地图继续验证；不能将基线退步全部归因为训练不足或把AB相对差距扩大当成AB自身都进步。
