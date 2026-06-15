# 小红书爆款预测器 (Streamlit · ModelScope Studio)

天津师范大学周边美食探店 · 发前/发后热度多模态预测系统

## 功能

| 模块 | 说明 |
|------|------|
| 发布前预测 | 输入标题/正文/标签, 走 5-fold RF 发前模型 |
| 发后重测 | 用户填首批评论后, 走发后模型 (F1=0.621) |
| 多模态融合 | 可选上传封面, ResNet50 + Chinese CLIP + 文本 RF 融合 |
| 图文匹配 | Chinese CLIP 相似度, 检测"图片与内容不符" |
| 改进建议 | 数据驱动的客观提示 (含特征相关性) |

## 模型架构

| 模块 | 算法 | F1 |
|------|------|-----|
| ① 文本分类器 (主) | 5-fold Random Forest 投票 + 类别权重平衡 | 发前 0.408 / 发后 0.621 |
| ② 图像分类器 (可选) | Fine-tuned ResNet50 (backbone 冻结) | 0.40 |
| ③ 图文对齐 (可选) | Chinese CLIP (OFA-Sys/chinese-clip-vit-base-patch16) | - |
| ④ 多模态融合 | 0.4 × ResNet50 + 0.6 × 文本 RF | 0.52 |

## 文件结构

```
app.py              # Streamlit 入口
predictor.py        # 文本 RF 预测 (V2 发前/发后)
predictor_cnn.py    # 多模态融合 (V3 GBM + ResNet50 + CLIP)
_compat.py          # numpy 版本兼容补丁 (V3 模型加载)
models/             # 12 个 .pkl/.pth/.json 模型文件
data/               # 3 个 .txt 词典
requirements.txt
```

## 训练数据

- 649 条天津师大周边美食探店帖子 (MediaCrawler 采集)
- 4000+ 评论 (情感分析训练)
- 738 张封面图片 (ResNet50 训练)

## 局限性

- 训练-推理分布偏移: 发前 POST 特征 (评论数等) 训练时 mask=0
- 649 条样本偏少, F1 上限约 0.65
- 情感词与热度相关性 r≈0.02 (情感≠热度)
- 训练样本全部是"已发布 ≥1 天"的帖子, 发前预测为诚实性能

## 技术栈

- Python 3.10+
- scikit-learn 1.5+ / Random Forest
- PyTorch 2.0+ / torchvision / ResNet50
- HuggingFace transformers / Chinese CLIP
- Streamlit 1.28+
