"""
小红书爆款预测器 — CNN迁移学习 + CLIP图文匹配检测

架构:
  1. Fine-tuned ResNet50 → 只看封面预测热度 (CNN)
  2. GBM (20维文本) → 只看文案预测热度 (Text)
  3. OpenAI CLIP → 封面与文案的语义相似度 (真正的图文对齐)
  4. CNN-Text 冲突 → 双模型预测冲突检测 (辅助)

V3 兼容性:
  - V3 GBM pickle 损坏 (numpy 1.x/2.x 混用), 加载失败时自动 fallback 到我的 V2 RF 模型
  - V2 RF (best_classifier_pre.pkl) 仍可正常加载 (numpy 1.26.4)
  - CNN/CLIP/PCA 不受影响

输出:
  - final_label: 综合预测 (CNN + Text 加权)
  - clip_similarity: CLIP 图文匹配分数
  - mismatch_warning: 图文不符警告
  - cnn_proba / text_proba: 各自的概率分布
"""
import joblib
import re
import os
import numpy as np

# ============== V3 兼容性补丁 (必须最先执行) ==============
try:
    from _compat import _patch_numpy_modules, _patch_numpy_pickle
    _patch_numpy_modules()
    _patch_numpy_pickle()
except Exception as e:
    print(f'[predictor_cnn] compat patch failed: {e}')

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ============== 路径 ==============
_HERE  = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(_HERE, 'models')
DATA   = os.path.join(_HERE, 'data')
COVERS = os.path.join(_HERE, '..', 'covers')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

# ============== CNN 模型 ==============
_cnn_model = None
_cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _load_cnn():
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    cnn_path = os.path.join(MODELS, 'finetuned_resnet50.pth')
    if not os.path.exists(cnn_path):
        print('[predictor_cnn] Fine-tuned CNN not found, using凍結ResNet50')
        return None

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(2048, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 3),
    )
    ckpt = torch.load(cnn_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE).eval()
    _cnn_model = model
    print(f'[predictor_cnn] Fine-tuned CNN loaded (Val F1={ckpt.get("val_f1", "?")})')
    return _cnn_model


def predict_from_cover(image: Image.Image) -> dict:
    """
    仅从封面图片预测热度 (CNN)

    Args:
        image: PIL RGB Image

    Returns:
        dict with label, proba
    """
    model = _load_cnn()
    if model is None:
        return {'label': -1, 'proba': [0.33, 0.33, 0.34], 'error': 'CNN not available'}

    img_tensor = _cnn_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(proba.argmax())
    return {
        'label': pred,
        'proba': proba.tolist(),
        'label_text': ['低热度', '中热度', '高热度'][pred],
    }


# ============== 文本 GBM 模型 (延迟加载, 避免 pickle 版本冲突) ==============
_v3_path = os.path.join(MODELS, 'classifier_pipeline_v3.pkl')
_text_pipe = None
_text_pipe_post = None  # 发后模型 (用 scaler_post)
_text_struct_cols = None
_text_struct_cols_post = None
_text_scaler = None
_text_fold_models = None
_text_post_idx = None
_text_word_idx = None
_text_chars_idx = None
_text_word_cap = None
_text_chars_cap = None
_text_f1 = 0.0
_text_img_dim = 16
_text_load_error = None

def _load_text_model():
    """延迟加载 GBM 文本模型"""
    global _text_pipe, _text_struct_cols, _text_scaler, _text_fold_models
    global _text_post_idx, _text_word_idx, _text_chars_idx, _text_word_cap, _text_chars_cap
    global _text_f1, _text_img_dim, _text_load_error

    # 检查所有必要的状态都就绪 (之前只检查 _text_pipe, 但 fold_models 加载失败时
    # _text_pipe 已被设值, _text_word_idx 等还是 None, 导致 V3 路径走到一半才崩)
    if (_text_pipe is not None and _text_fold_models is not None
            and _text_word_idx is not None and _text_word_cap is not None):
        return True
    if _text_load_error is not None:
        return False

    if not os.path.exists(_v3_path):
        _text_load_error = 'classifier_pipeline_v3.pkl not found'
        print(f'[predictor_cnn] WARNING: {_text_load_error}')
        return False

    try:
        _text_pipe = joblib.load(_v3_path)
        _text_struct_cols = _text_pipe['structured_cols']
        _text_scaler = _text_pipe['scaler_pre']
        _text_bundle = joblib.load(os.path.join(MODELS, 'best_classifier_pre_img.pkl'))
        _text_fold_models = _text_bundle['fold_models']
        _text_post_idx = _text_pipe['post_idx']
        _text_word_idx = _text_pipe['word_idx']
        _text_chars_idx = _text_pipe['chars_idx']
        _text_word_cap = _text_pipe['word_cap']
        _text_chars_cap = _text_pipe['chars_cap']
        _text_f1 = _text_pipe['f1_pre']
        _text_img_dim = _text_pipe.get('img_pca_dim', 16)
        print(f'[predictor_cnn] Text GBM loaded (F1={_text_f1:.3f})')
        return True
    except Exception as e:
        # 加载失败: 重置所有相关全局, 避免下次误判为已加载
        _text_pipe = None
        _text_fold_models = None
        _text_word_idx = None
        _text_chars_idx = None
        _text_word_cap = None
        _text_chars_cap = None
        _text_post_idx = None
        _text_scaler = None
        _text_struct_cols = None
        _text_load_error = str(e)
        print(f'[predictor_cnn] Text model load failed: {e}')
        return False

# ============== 文本 V2 Fallback (Raspberry / 加载失败时用) ==============
# 与 deploy-v5/predictor.py 一致: 加 logit/T + decision_bias calibration,
# 解决"高热度偏向"问题
_text_temperature = None
_text_decision_bias = None
def _ensure_calibration_loaded():
    global _text_temperature, _text_decision_bias
    if _text_temperature is not None:
        return
    if _text_pipe is None:
        _text_temperature = 1.0
        _text_decision_bias = np.array([0.0, 0.0, 0.0])
        return
    _text_temperature = float(_text_pipe.get('temperature', 1.0))
    _text_decision_bias = np.array(_text_pipe.get('decision_bias', [0.0, 0.0, 0.0]))

# ============== 情感 + 词典 (延迟加载) ==============
import pandas as pd
import jieba

_sent_pipe = None
_sent_tfidf = None
_sent_clf = None
_sent_load_error = None

def _load_sentiment():
    """延迟加载情感模型"""
    global _sent_pipe, _sent_tfidf, _sent_clf, _sent_load_error
    if _sent_clf is not None:
        return True
    if _sent_load_error is not None:
        return False
    try:
        _sent_pipe = joblib.load(os.path.join(MODELS, 'sentiment_model.pkl'))
        _sent_tfidf = joblib.load(os.path.join(MODELS, 'sentiment_tfidf.pkl'))
        _sent_clf = _sent_pipe['clf']
        return True
    except Exception as e:
        _sent_load_error = str(e)
        print(f'[predictor_cnn] Sentiment model load failed: {e}')
        return False

def _load_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(w.strip() for w in f if w.strip())

_stopwords = None
_pos_words = None
_neg_words = None

def _ensure_lexicons():
    global _stopwords, _pos_words, _neg_words
    if _stopwords is None:
        _stopwords = _load_words(os.path.join(DATA, 'stopwords.txt'))
        _pos_words = _load_words(os.path.join(DATA, 'positive_words.txt'))
        _neg_words = _load_words(os.path.join(DATA, 'negative_words.txt'))

def _clean_text(s):
    _ensure_lexicons()
    # 防御: 不管传进来什么 (None/float/int/list), 强转成字符串
    if s is None:
        s = ''
    else:
        try:
            s = str(s)
        except Exception:
            s = ''
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', s)
    # 防御: jieba 偶尔会返回非字符串 token (实测过几次), 强制过滤
    result = []
    for w in jieba.cut(s):
        if not isinstance(w, str):
            continue
        try:
            if w.strip() and w not in (_stopwords or set()) and len(w) > 1:
                result.append(w)
        except Exception:
            continue
    return ' '.join(result)

# ============== V2 Fallback 文本预测 ==============
_v2_pipe = None
_v2_fold_models_pre = None
_v2_load_error = None

def _load_v2_text_model():
    """加载我的 V2 RF 模型 (作为 V3 GBM 失败时的 fallback)"""
    global _v2_pipe, _v2_fold_models_pre, _v2_load_error
    if _v2_fold_models_pre is not None:
        return True
    if _v2_load_error is not None:
        return False
    try:
        v2_path = os.path.join(MODELS, 'classifier_pipeline_v2.pkl')
        if not os.path.exists(v2_path):
            _v2_load_error = f'{v2_path} not found'
            return False
        _v2_pipe = joblib.load(v2_path)
        bundle = joblib.load(os.path.join(MODELS, 'best_classifier_pre.pkl'))
        _v2_fold_models_pre = bundle['fold_models']
        print(f'[predictor_cnn] V2 RF fallback loaded (F1={_v2_pipe["f1_pre"]:.3f})')
        return True
    except Exception as e:
        _v2_load_error = str(e)
        print(f'[predictor_cnn] V2 RF fallback load failed: {e}')
        return False

def _predict_from_text_v2_fallback(title: str, content: str, tags: str = '') -> dict:
    """V2 RF 模型推理 — 直接复用 predict_with_mode (它已包含特征+transform)"""
    # 防御: 同 predict_from_text
    title = str(title) if title is not None else ''
    content = str(content) if content is not None else ''
    tags = str(tags) if tags is not None else ''
    if not _load_v2_text_model():
        return {'label': -1, 'proba': [0.33, 0.33, 0.34], 'error': _v2_load_error or 'No model available'}
    from predictor import predict_with_mode
    res = predict_with_mode(title, content, tags, None, mode='pre')
    return {
        'label': res['label'],
        'proba': res['proba'],
        'label_text': res.get('label_text', ['低', '中', '高'][res['label']]),
        'note': 'V2 RF fallback (V3 GBM 加载失败)',
    }

def predict_from_text(title: str, content: str, tags: str = '') -> dict:
    """仅从文本预测热度 (GBM V3, 失败时回退到 V2 RF)"""
    # 防御: 任何参数都可能因为 session_state 同步异常变成 None, 强转 str
    title = str(title) if title is not None else ''
    content = str(content) if content is not None else ''
    tags = str(tags) if tags is not None else ''
    if not _load_text_model():
        # V3 加载失败, 回退到我的 V2 RF 模型
        return _predict_from_text_v2_fallback(title, content, tags)
    # 防御: 加载成功但状态不全 (旧 bug) → 仍走 V2
    if (_text_word_idx is None or _text_word_cap is None
            or _text_fold_models is None or _text_scaler is None):
        print(f'[predictor_cnn] V3 状态不全, 走 V2 fallback')
        return _predict_from_text_v2_fallback(title, content, tags)
    _ensure_lexicons()

    # 提取20维文本特征
    title_clean = _clean_text(title)
    content_clean = _clean_text(content)

    raw = {
        'title_len_chars': len(str(title)),
        'content_len_chars': len(str(content)),
        'title_len_words': len(title_clean.split()),
        'content_len_words': len(content_clean.split()),
        'tag_count': max(len([t for t in tags.split(',') if t.strip()]),
                         len([t for t in tags.split('#') if t.strip()])) if tags.strip() else 0,
        'days_alive': 176.0,
        'comment_n': 0.0, 'cmt_avg_len': 0.0,
        'title_sent': len(set(jieba.cut(str(title))) & _pos_words) - len(set(jieba.cut(str(title))) & _neg_words),
        'pos_ratio': 0.0, 'neg_ratio': 0.0, 'neutral_ratio': 0.0, 'sent_strength': 0.0,
        # 代理特征
        'tag_has_food': int(bool(re.search(r'美食|探店|好吃|必吃|推荐|餐厅|小吃|火锅|烧烤',
                                            tags.replace('#',' ').replace('，',',').replace(',',' ')))),
        'content_lines': str(content).count('\n') + 1,
        'content_has_loc': int(bool(re.search(r'天师|大|师大|校园|南开|河西|理工', str(content)))),
        'content_digit_ratio': sum(c.isdigit() for c in str(content)) / max(len(str(content)), 1),
        'content_has_price': int(bool(re.search(r'\d+元|\d+块|人均|\d+¥', str(content)))),
        'title_has_q': int(bool(re.search(r'\?|？', str(title)))),
        'content_avg_word': float(np.mean([len(w) for w in str(content)])) if str(content) else 0.0,
    }

    X_raw = np.array([[raw[c] for c in _text_struct_cols]], dtype=float)
    # log1p cap
    if isinstance(_text_word_idx, (list, np.ndarray)):
        for i in _text_word_idx:
            X_raw[0, i] = np.log1p(min(X_raw[0, i], _text_word_cap))
        for i in _text_chars_idx:
            X_raw[0, i] = np.log1p(min(X_raw[0, i], _text_chars_cap))
    else:
        X_raw[0, _text_word_idx] = np.log1p(min(X_raw[0, _text_word_idx], _text_word_cap))
        X_raw[0, _text_chars_idx] = np.log1p(min(X_raw[0, _text_chars_idx], _text_chars_cap))
    # mask POST (发前场景: 强制 mask, 不依赖训练是否学到)
    for idx in _text_post_idx:
        X_raw[0, idx] = 0

    # V3 = 20 维 (与 V2 同结构, 不加图像占位)
    X_std = _text_scaler.transform(X_raw)

    proba_list = [m.predict_proba(X_std)[0] for m in _text_fold_models]
    proba = np.mean(proba_list, axis=0)

    # Calibration (与 deploy-v5 一致): logit/T + bias, 解决"高热度偏向"
    _ensure_calibration_loaded()
    logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6)) / _text_temperature + _text_decision_bias
    # 修复: 返回校准后的 proba (softmax 后的 logit), 让 proba 和 label_text 一致
    exp_logit = np.exp(logit - logit.max())
    proba_cal = (exp_logit / exp_logit.sum()).tolist()
    pred = int(np.argmax(proba_cal))
    return {
        'label': pred,
        'proba': proba_cal,  # 校准后的 proba, 跟 label_text 一致
        'proba_raw': proba.tolist(),  # 保留 raw 供调试
        'label_text': ['低热度', '中热度', '高热度'][pred],
    }


# ============== CLIP 图文语义对齐 ==============
_clip_model = None
_clip_processor = None

# Load pre-computed CLIP threshold
import json
_clip_threshold_path = os.path.join(MODELS, 'clip_threshold.json')
if os.path.exists(_clip_threshold_path):
    with open(_clip_threshold_path) as f:
        _clip_stats = json.load(f)
    _clip_threshold = _clip_stats['threshold']
    ns = _clip_stats.get('n_samples', '?')
    print(f'[predictor_cnn] CLIP threshold loaded: {_clip_threshold:.1f} (from {ns} samples)')
else:
    _clip_threshold = 36.0  # Fallback
    _clip_stats = None
    print(f'[predictor_cnn] CLIP threshold using fallback: {_clip_threshold:.1f}')

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel
    print('[predictor_cnn] Loading Chinese CLIP (OFA-Sys/chinese-clip-vit-base-patch16)...')
    _clip_model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    _clip_processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    _clip_model.eval()
    for p in _clip_model.parameters():
        p.requires_grad = False
    return _clip_model, _clip_processor


def compute_clip_similarity(image: Image.Image, title: str, content: str) -> float:
    """
    用 OpenAI CLIP 计算封面图片与文本的语义相似度
    返回原始 logit 值 (越高越匹配)
    """
    try:
        model, processor = _load_clip()
        text = f'{title} {content[:80]}'[:120]  # Chinese CLIP: 52 token limit, truncate chars
        inputs = processor(text=[text], images=image, return_tensors='pt', padding=True, truncation=True, max_length=52)
        with torch.no_grad():
            outputs = model(**inputs)
            sim = outputs.logits_per_image[0][0].item()
        return sim
    except Exception as e:
        print(f'[predictor_cnn] CLIP error: {e}')
        return None


# ============== 综合预测 + 图文匹配检测 ==============
def predict_combined(title: str, content: str, tags: str = '',
                     image: Image.Image = None,
                     mode: str = 'pre', comments: list = None) -> dict:
    """
    融合 CNN + GBM 的综合预测，含 CLIP 图文对齐检测

    参数:
      mode='pre':  用发前文本模型 (V3 GBM pre 失败→V2 RF pre fallback)
      mode='post': 用发后文本模型 (我的 V2 RF post, 支持评论聚合)
      comments:   发后模式下的评论列表 (聚合情感特征)

    检测逻辑:
      - CLIP 计算封面与文本的语义相似度
      - CNN(封面) 和 GBM(文本) 分别预测热度
      - CLIP低分 + 模型冲突 → 严重图文不符

    Returns:
        dict with:
          - final_label, final_proba: 综合结果
          - clip_similarity: CLIP 图文匹配分数
          - cnn_result, text_result: 各自的预测
          - mismatch: bool
          - mismatch_level: 'none' | 'mild' | 'severe'
          - mismatch_reason: str
          - suggestions: list
    """
    # 防御: session_state 同步异常可能让任何字段变成 None, 强转 str
    title = str(title) if title is not None else ''
    content = str(content) if content is not None else ''
    tags = str(tags) if tags is not None else ''
    if comments is None:
        comments = []
    global _clip_threshold
    results = {}

    # CNN 预测
    if image is not None:
        cnn = predict_from_cover(image)
        results['cnn'] = cnn
    else:
        cnn = None
        results['cnn'] = None

    # Text 预测 (发后模式: 用我 V2 post 支持评论; 发前模式: 用 V3/V2 pre)
    if mode == 'post':
        from predictor import predict_with_mode
        text = predict_with_mode(title, content, tags, comments or [], mode='post')
        text = {
            'label': text['label'],
            'proba': text['proba'],
            'label_text': text.get('label_text', ['低热度', '中热度', '高热度'][text['label']]),
        }
    else:
        text = predict_from_text(title, content, tags)
    results['text'] = text

    # 综合预测
    if cnn is not None and cnn.get('error') is None:
        # CNN + Text 加权融合 (CNN权重 0.4, Text权重 0.6 — Text F1 更高)
        cnn_proba = np.array(cnn['proba'])
        text_proba = np.array(text['proba'])
        fused_proba = 0.4 * cnn_proba + 0.6 * text_proba
        final_label = int(fused_proba.argmax())
        results['fusion_weight'] = '0.4xCNN + 0.6xText'
    else:
        fused_proba = np.array(text['proba'])
        final_label = text['label']
        results['fusion_weight'] = 'Text only (no image)'

    results['final_label'] = final_label
    results['final_proba'] = fused_proba.tolist()
    results['final_label_text'] = ['低热度', '中热度', '高热度'][final_label]

    # ---- 图文匹配检测 ----
    mismatch_level = 'none'
    mismatch_reason = ''

    # 1. CLIP 语义对齐检测
    clip_sim = None
    if image is not None:
        clip_sim = compute_clip_similarity(image, title, content)

    if cnn is not None and cnn.get('error') is None:
        cnn_label = cnn['label']
        text_label = text['label']
        label_diff = abs(cnn_label - text_label)

        clip_low = clip_sim is not None and clip_sim < _clip_threshold

        if clip_low and label_diff >= 1:
            # CLIP says low similarity AND models disagree → severe
            mismatch_level = 'severe'
            mismatch_reason = (
                f'CLIP 图文匹配分数异常低 ({clip_sim:.1f} < 阈值{_clip_threshold:.0f}), '
                f'且封面预测为"{cnn["label_text"]}", 文案预测为"{text["label_text"]}"。'
                f'封面图片与文案内容可能严重不符, 建议更换封面或修改文案。'
            )
        elif clip_low:
            # CLIP only
            mismatch_level = 'mild'
            mismatch_reason = (
                f'CLIP 图文匹配分数偏低 ({clip_sim:.1f} < 阈值{_clip_threshold:.0f})。'
                f'封面图片可能与文案主题不相关, 建议检查。'
            )
        elif label_diff >= 2:
            # 方案 A: 单独的 label_diff>=2 不再 severe, 因为 941 样本下 F1=0.4 模型预测跳 2 档是正常波动
            # 只在 CLIP 同时也低时才 severe
            mismatch_level = 'mild'
            mismatch_reason = (
                f'封面预测为"{cnn["label_text"]}", 但文案预测为"{text["label_text"]}", 两者差距较大。'
                f'考虑到 941 条训练样本下 F1 仅 0.4~0.5, 此差异可能是模型波动, 不一定是图文不符。'
                f'若要严格判断, 请参考 CLIP 图文匹配分: {_clip_threshold-10 if _clip_threshold else "未计算"} 以上较可靠。'
            )
        elif label_diff == 1:
            mismatch_level = 'mild'
            mismatch_reason = (
                f'封面预测为"{cnn["label_text"]}", 文案预测为"{text["label_text"]}"。'
                f'封面和文案信号略有偏差, 可考虑微调标题或封面使其更一致。'
            )
    elif image is not None and clip_sim is not None:
        # Only CLIP available
        if _clip_threshold is None:
            _clip_threshold = 20.0
        if clip_sim < _clip_threshold:
            mismatch_level = 'mild'
            mismatch_reason = f'CLIP 图文匹配分数偏低 ({clip_sim:.1f})。建议检查封面是否与内容相关。'

    results['clip_similarity'] = clip_sim
    results['clip_threshold'] = _clip_threshold
    results['mismatch'] = mismatch_level != 'none'
    results['mismatch_level'] = mismatch_level
    results['mismatch_reason'] = mismatch_reason

    # ---- 改进建议 ----
    results['suggestions'] = _build_suggestions(
        title, content, tags, final_label, results
    )

    return results


def _build_suggestions(title, content, tags, final_label, results):
    _ensure_lexicons()
    suggestions = []

    # 文本特征建议
    chars = len(str(content))
    if chars < 100:
        suggestions.append(f'正文偏短 ({chars} 字), 建议扩充至 200~400 字')
    elif chars > 800:
        suggestions.append(f'正文过长 ({chars} 字), 建议精简到 300~500 字')

    sent = len(set(jieba.cut(str(title))) & (_pos_words or set())) - len(set(jieba.cut(str(title))) & (_neg_words or set()))
    if sent <= 0:
        suggestions.append('标题情感强度不足, 多用"绝了、巨好吃、必冲"等强烈词')

    tag_count = max(len([t for t in tags.split(',') if t.strip()]),
                    len([t for t in tags.split('#') if t.strip()])) if tags.strip() else 0
    if tag_count < 3:
        suggestions.append(f'标签偏少 ({tag_count} 个), 建议加到 3~5 个')

    if '！' not in title and '!' not in title:
        suggestions.append('标题缺少感叹号, 加一个能提升情绪感染力')

    # 图文不符警告
    if results['mismatch_level'] == 'severe':
        suggestions.insert(0, f'[图文不符警告] {results["mismatch_reason"]}')
    elif results['mismatch_level'] == 'mild':
        suggestions.insert(0, f'[注意] {results["mismatch_reason"]}')

    # 封面建议
    if results.get('cnn') and results['cnn'].get('error') is None:
        cnn_label = results['cnn']['label']
        if cnn_label == 0:
            suggestions.append('封面视觉吸引力偏弱, 建议使用高饱和度、清晰构图、食物特写的封面')
        elif cnn_label == 2:
            suggestions.append('封面视觉效果很好, 保持这个风格!')

    if final_label == 2 and not suggestions:
        suggestions.append('各项指标都很好, 封面和文案匹配度也高, 直接发布!')
    elif final_label == 0 and not suggestions:
        suggestions.append('综合特征偏弱, 建议优化正文长度/标题情感/标签数量, 并检查封面质量')

    return suggestions
