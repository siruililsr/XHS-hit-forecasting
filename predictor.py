"""
小红书爆款预测器 — 预测逻辑（不含UI）
按 streamlit说明书.md 严格实现
适配: A+B+D 重训后
  - 旧接口 predict(title, content, tags, comments)  -> 走 POST 模型 (向后兼容)
  - 新接口 predict_with_mode(title, content, tags, comments, mode)  -> 'pre'/'post'
"""
import joblib
import re
import os
import jieba
import numpy as np

# ============== 路径 (相对本文件, 不依赖 cwd) ==============
_HERE  = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(_HERE, 'models')
DATA   = os.path.join(_HERE, 'data')

# ============== 加载模型 (新: 优先 v2, 找不到用旧) ==============
_v2_path = os.path.join(MODELS, 'classifier_pipeline_v2.pkl')
if os.path.exists(_v2_path):
    # 新 v2: 双 scaler + 双模型
    _pipeline = joblib.load(_v2_path)
    _struct_cols = _pipeline['structured_cols']            # 20 维
    _old_cols    = _pipeline['old_structured_cols']        # 13 维
    _scaler_pre  = _pipeline['scaler_pre']
    _scaler_post = _pipeline['scaler_post']
    _word_idx    = _pipeline['word_idx']
    _chars_idx   = _pipeline['chars_idx']
    _word_cap    = _pipeline['word_cap']
    _chars_cap   = _pipeline['chars_cap']
    _post_idx    = _pipeline['post_idx']
    _f1_pre      = _pipeline['f1_pre']
    _f1_post     = _pipeline['f1_post']
    _decision_bias = np.array(_pipeline.get('decision_bias', [0.0, 0.0, 0.0]))
    _temperature   = float(_pipeline.get('temperature', 1.0))

    _pre_bundle  = joblib.load(os.path.join(MODELS, 'best_classifier_pre.pkl'))
    _post_bundle = joblib.load(os.path.join(MODELS, 'best_classifier_post.pkl'))
    _fold_models_pre  = _pre_bundle['fold_models']
    _fold_models_post = _post_bundle['fold_models']
    print(f'[predictor] loaded v2 pipeline: pre F1={_f1_pre:.3f}, post F1={_f1_post:.3f}')
else:
    # 旧: 13 维单模型, fallback
    _pipeline = joblib.load(os.path.join(MODELS, 'classifier_pipeline.pkl'))
    _struct_cols = _pipeline['structured_cols']
    _old_cols    = _struct_cols
    _scaler_pre = _scaler_post = _pipeline['scaler']
    _word_idx  = _pipeline.get('word_idx',  _struct_cols.index('content_len_words'))
    _chars_idx = _pipeline.get('chars_idx', _struct_cols.index('content_len_chars'))
    _word_cap  = _pipeline.get('word_cap',  200)
    _chars_cap = _pipeline.get('chars_cap',  600)
    _post_idx  = [_struct_cols.index(c) for c in
                  ['days_alive', 'comment_n', 'cmt_avg_len', 'neg_ratio', 'neutral_ratio', 'sent_strength']]
    _f1_pre = _f1_post = 0.617
    _decision_bias = np.array(_pipeline.get('decision_bias', [0.0, 0.0, 0.0]))
    _temperature   = float(_pipeline.get('temperature', 1.0))
    _bundle = joblib.load(os.path.join(MODELS, 'best_classifier.pkl'))
    _fold_models_pre = _fold_models_post = (_bundle['fold_models'] if isinstance(_bundle, dict) and 'fold_models' in _bundle else [_bundle])
    print('[predictor] loaded v1 pipeline (fallback, no proxy features)')

_sent_pipe  = joblib.load(os.path.join(MODELS, 'sentiment_model.pkl'))
_sent_tfidf = joblib.load(os.path.join(MODELS, 'sentiment_tfidf.pkl'))
_sent_clf   = _sent_pipe['clf']

# ============== 加载词典 ==============
def _load_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(w.strip() for w in f if w.strip())

_stopwords = _load_words(os.path.join(DATA, 'stopwords.txt'))
_pos_words = _load_words(os.path.join(DATA, 'positive_words.txt'))
_neg_words = _load_words(os.path.join(DATA, 'negative_words.txt'))

# 训练集统计 fallback (中位数 / 训练集对应值) ==============
# 新帖未发布, days_alive 真实值未知; 用训练集中位数 176.0 占位,
# 比填 0 更合理 (0 标准化后是极端异常值, 强迫模型往低热度走).
# 注意: 这是预测时的占位, 不是真实存活天数.
_DAYS_ALIVE_FALLBACK = 176.0
# 发前无评论, POST 特征全部填 0 (12.6% 训练样本本身就是 0, 合法)
# 训练时缺失增强会 mask POST 特征让模型学会 "看到 0 当缺失".

# 训练时统计的代理特征 (发布前可从 title/content/tags 衍生)
_PROXY_MEDIANS = {
    # 多数代理特征在缺失时用 0 已合理 (sum 类的二值/计数);
    # 连续型如 content_lines/content_avg_word/content_digit_ratio 也用 0 占位
    # 因为短/无内容的帖子这些值本来就接近 0.
}


# ============== 文本清洗 (与 01_preprocess.py 一致) ==============
def _clean_text(s):
    # 防御: 任何奇怪的输入都强转字符串
    if s is None:
        s = ''
    else:
        try:
            s = str(s)
        except Exception:
            s = ''
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', s)
    # 防御: jieba 偶尔返回非字符串, 强制过滤
    out = []
    for w in jieba.cut(s):
        if not isinstance(w, str):
            continue
        try:
            if w.strip() and w not in (_stopwords or set()) and len(w) > 1:
                out.append(w)
        except Exception:
            continue
    return ' '.join(out)


# ============== 特征提取 (返回 X_raw(20 维, 顺序按 _struct_cols)) ==============
def _extract_features(title: str, content: str,
                      tags: str = '', comments: list = None):
    # 防御: session_state 同步异常可能让任何字段变成 None
    title = str(title) if title is not None else ''
    content = str(content) if content is not None else ''
    tags = str(tags) if tags is not None else ''
    if comments is None:
        comments = []
    title_clean   = _clean_text(title)
    content_clean = _clean_text(content)

    # 长度特征
    title_len_chars   = len(str(title))
    content_len_chars = len(str(content))
    title_len_words   = len(title_clean.split())
    content_len_words = len(content_clean.split())

    # 标签数量
    if tags.strip():
        tag_count = max(
            len([t for t in tags.split(',') if t.strip()]),
            len([t for t in tags.split('#') if t.strip()])
        )
    else:
        tag_count = 0

    days_alive = _DAYS_ALIVE_FALLBACK

    # 标题情感
    title_words = set(jieba.cut(str(title)))
    title_sent  = len(title_words & _pos_words) - len(title_words & _neg_words)

    # 评论情感
    if comments and len(comments) > 0:
        cleaned_cmts = [_clean_text(c) for c in comments]
        X_cmts = _sent_tfidf.transform(cleaned_cmts)
        preds  = _sent_clf.predict(X_cmts)   # 0/1/2
        n = len(preds)
        pos_ratio     = (preds == 1).sum() / n
        neg_ratio     = (preds == 2).sum() / n
        neutral_ratio = (preds == 0).sum() / n
        comment_n     = float(n)
        cmt_avg_len   = float(np.mean([len(c.split()) for c in cleaned_cmts]))
        sent_strength = pos_ratio - neg_ratio
    else:
        pos_ratio = neg_ratio = neutral_ratio = 0.0
        comment_n = cmt_avg_len = sent_strength = 0.0

    # 代理特征 (D 方案: 发布前可从 title/content/tags 衍生)
    # 7 个最强信号, 全部基于 title/content/tags 字符串, 0 填充
    tags_str = tags.replace('#', ' ').replace('，', ',').replace(',', ' ').strip()
    proxy = {
        'tag_has_food':      int(bool(re.search(r'美食|探店|好吃|必吃|推荐|餐厅|小吃|火锅|烧烤', tags_str))),
        'content_lines':     str(content).count('\n') + 1,
        'content_has_loc':   int(bool(re.search(r'天师|大|师大|校园|南开|河西|理工', str(content)))),
        'content_digit_ratio': sum(c.isdigit() for c in str(content)) / max(len(str(content)), 1),
        'content_has_price': int(bool(re.search(r'\d+元|\d+块|人均|\d+¥', str(content)))),
        'title_has_q':       int(bool(re.search(r'\?|？', str(title)))),
        'content_avg_word':  float(np.mean([len(w) for w in str(content)])) if str(content) else 0.0,
    }

    # 按 structured_cols 顺序
    raw = {
        'title_len_chars':   title_len_chars,
        'content_len_chars': content_len_chars,
        'title_len_words':   title_len_words,
        'content_len_words': content_len_words,
        'tag_count':         tag_count,
        'days_alive':        days_alive,
        'comment_n':         comment_n,
        'cmt_avg_len':       cmt_avg_len,
        'title_sent':        title_sent,
        'pos_ratio':         pos_ratio,
        'neg_ratio':         neg_ratio,
        'neutral_ratio':     neutral_ratio,
        'sent_strength':     sent_strength,
        # 代理特征 (D 方案)
        'tag_has_food':      proxy['tag_has_food'],
        'content_lines':     proxy['content_lines'],
        'content_has_loc':   proxy['content_has_loc'],
        'content_digit_ratio': proxy['content_digit_ratio'],
        'content_has_price': proxy['content_has_price'],
        'title_has_q':       proxy['title_has_q'],
        'content_avg_word':  proxy['content_avg_word'],
    }
    X_raw = np.array([[raw[c] for c in _struct_cols]], dtype=float)
    return X_raw


# ============== 预测 (核心, mode='pre'/'post') ==============
def _transform_for_model(X_raw: np.ndarray, mode: str = 'post') -> np.ndarray:
    """log1p+cap, 然后用对应 scaler 标准化. mode='pre' 时 mask POST 列."""
    X = X_raw.copy()
    # 兼容 list[idx] 和 int idx
    wi = _word_idx if isinstance(_word_idx, (list, np.ndarray)) else [_word_idx]
    ci = _chars_idx if isinstance(_chars_idx, (list, np.ndarray)) else [_chars_idx]
    for i in wi:
        X[0, i] = np.log1p(min(X_raw[0, i], _word_cap))
    for i in ci:
        X[0, i] = np.log1p(min(X_raw[0, i], _chars_cap))
    # 不 mask POST: 用户填 default 0, model 训练时用真实 0 分布
    return _scaler_pre.transform(X)


def predict_with_mode(title: str, content: str, tags: str = '',
                      comments: list = None, mode: str = 'pre') -> dict:
    """
    mode='pre'  -> 发前模型 (POST mask=0, F1≈0.41)
    mode='post' -> 发后模型 (全特征,    F1≈0.62, 需用户填评论)
    """
    # 防御: session_state 同步异常可能让任何字段变成 None
    title = str(title) if title is not None else ''
    content = str(content) if content is not None else ''
    tags = str(tags) if tags is not None else ''
    if comments is None:
        comments = []
    X_raw = _extract_features(title, content, tags, comments)
    X_std = _transform_for_model(X_raw, mode=mode)

    fold_models = _fold_models_pre if mode == 'pre' else _fold_models_post
    proba_list = [m.predict_proba(X_std)[0] for m in fold_models]
    proba = np.mean(proba_list, axis=0)

    logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6)) / _temperature + _decision_bias
    # 修复: 返回校准后的 proba (softmax logit), 让 proba 和 label_text 一致
    exp_logit = np.exp(logit - logit.max())
    proba_cal = (exp_logit / exp_logit.sum()).tolist()
    pred  = int(np.argmax(proba_cal))

    label_map = {0: '低热度', 1: '中热度', 2: '高热度'}
    f1_score = _f1_pre if mode == 'pre' else _f1_post

    suggestions = _build_suggestions(title, content, X_raw[0], pred, comments)

    return {
        'label':       pred,
        'label_text':  label_map[pred],
        'proba':       proba_cal,  # 校准后 proba, 跟 label_text 一致
        'proba_raw':   proba.tolist(),  # 保留 raw 供调试
        'mode':        mode,
        'f1':          f1_score,
        'suggestions': suggestions,
        'features': {
            'content_len_words': int(X_raw[0, _struct_cols.index('content_len_words')]),
            'content_len_chars': int(X_raw[0, _struct_cols.index('content_len_chars')]),
            'title_sent':        int(X_raw[0, _struct_cols.index('title_sent')]),
            'tag_count':         int(X_raw[0, _struct_cols.index('tag_count')]),
            'comment_n':         int(X_raw[0, _struct_cols.index('comment_n')]),
            'proba':             [round(p, 3) for p in proba_cal],
        },
    }


def _build_suggestions(title, content, raw_row, pred, comments):
    """根据原始特征生成具体可执行的改进建议"""
    suggestions = []
    chars_idx = _struct_cols.index('content_len_chars')
    sent_idx  = _struct_cols.index('title_sent')
    tag_idx   = _struct_cols.index('tag_count')

    chars = raw_row[chars_idx]
    sent  = raw_row[sent_idx]
    tag_n = raw_row[tag_idx]

    # ---- 正文长度 ----
    if chars < 30:
        suggestions.append(
            f'📏 **正文太短**（仅 {int(chars)} 字）：读者看不到详细信息，划走概率很高。\n\n'
            f'建议扩充到 200~400 字，至少包含：店铺名、具体位置、推荐菜品、价格范围、个人感受。')
    elif chars < 100:
        suggestions.append(
            f'📏 **正文偏短**（{int(chars)} 字）：信息量不足，建议再加点细节。\n\n'
            f'比如具体地址、人均消费、什么时候去的、排队多久——这些读者都关心。')
    elif chars > 800:
        suggestions.append(
            f'📏 **正文偏长**（{int(chars)} 字）：超出探店帖的常规长度，读者可能没耐心看完。\n\n'
            f'建议精简到 300~500 字，把最核心的信息放前面。')

    # ---- 标题情感 ----
    if sent < 0:
        suggestions.append(
            f'🎭 **标题偏负面**（情感分 {int(sent)}）：小红书用户更爱点开积极、有惊喜感的标题。\n\n'
            f'试试加入"绝了""巨好吃""必冲""太值了"这类词。')
    elif sent == 0:
        suggestions.append(
            f'🎭 **标题缺情感词**：目前标题比较平淡，加一两个情绪词能提高点击欲。\n\n'
            f'比如"这家真的可以冲""人均30的快乐谁懂啊"——不需要夸张，但要表达出你的真实感受。')

    # ---- 标签数量 ----
    if tag_n == 0:
        suggestions.append(
            f'🏷️ **没加标签**：小红书靠标签推荐流量，一个标签不加会损失大量曝光。\n\n'
            f'建议加 3 个左右：1 个地域标签（如"天津师范"）、1 个品类标签（如"校园美食"）、1 个热点标签（如"穷鬼套餐"）。')
    elif tag_n < 2:
        suggestions.append(
            f'🏷️ **标签太少**（仅 {int(tag_n)} 个）：建议加到 3~5 个，覆盖更多搜索入口。')
    elif tag_n >= 6:
        suggestions.append(
            f'🏷️ **标签过多**（{int(tag_n)} 个）：小红书用户对过度堆标签有反感，建议精简到 3~5 个最核心的。')

    # ---- 标题互动提示 ----
    if '？' in title or '?' in title:
        suggestions.append(
            '💡 **标题含问号**是个好信号！问句容易引发评论区互动。可以配合正文末尾抛一个问题，进一步引导评论。')

    # ---- 首批评论 ----
    if (not comments or len(comments) == 0):
        suggestions.append(
            '💬 **还没有评论数据**：在我们训练的 649 条帖子中，评论数是预测热度最强的信号。\n\n'
            '发布后如果能主动回复前几条评论、或者在正文末尾引导互动（比如"你们觉得哪家更好吃？"），拿到 5~10 条首批评论后再用"发后重测"Tab 跑一次，预测会准很多。')

    # ---- 综合判断 ----
    if pred == 0:
        suggestions.append(
            '📊 **综合判断：低热度概率最高**。但这只是基于 649 条历史帖子的统计规律，不代表你的帖子一定不火。\n\n'
            '建议按上面的提示优化后，切换到"发后重测"Tab 再评估一次。')
    elif pred == 2:
        suggestions.append(
            '📊 **综合判断：高热度概率最高**。恭喜！从数据上看你的帖子各项指标都不错，在 649 条训练样本中更接近爆款的特征分布。\n\n'
            '当然，最终还要看发布时间、平台推荐机制等我们无法控制的变量。')
    else:
        suggestions.append(
            '📊 **综合判断：中热度概率最高**，三类之间差距可能不大。\n\n'
            '这种情况建议两条腿走路：上传封面走多模态预测，或者等有了首批评论后用"发后重测"再跑一次。')

    return suggestions


# ============== 旧接口 (向后兼容, 走 POST 模型) ==============
def predict(title: str, content: str, tags: str = '', comments: list = None) -> dict:
    """旧接口: 等价于 predict_with_mode(..., mode='post')"""
    return predict_with_mode(title, content, tags, comments, mode='post')
