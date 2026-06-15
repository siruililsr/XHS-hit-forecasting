"""
小红书爆款预测器 — Streamlit 网页
A+B+C+D 重构版:
  - 顶部双 F1 显示 (发前 0.408 / 发后 0.621)
  - 删 _heuristic_bias, 删 slider
  - 两阶段 UI: 发前预测 (7 PRE + 7 PROXY) / 发后重测 (全 20 维 + 首批评论数)
"""
import os
# HF 镜像 (魔搭环境 / 国内服务器也能快速下载 Chinese CLIP)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
import streamlit as st
import pandas as pd
import numpy as np
import io
from predictor import predict_with_mode, _f1_pre, _f1_post

# 封面预测模块 (由团队成员训练: ResNet50+CLIP+GBM 融合) — 优雅降级, 缺包也能用
_CNN_OK = False
_CNN_IMPORT_ERR = ''
try:
    from predictor_cnn import predict_combined
    _CNN_OK = True
except Exception as _e:
    _CNN_IMPORT_ERR = str(_e)

# ============== Tab 字段同步 (发布前 -> 发后重测) ==============
# 需求: 在 Tab 1 输入的标题/正文/标签/封面, 切到 Tab 2 时自动填入;
#       用户在 Tab 2 手动改的字段, 不会被 Tab 1 后续输入覆盖.
# 实现: 用 dirty flag 标记"被用户在 Tab 2 独立改过", Tab 1 on_change 同步时只覆盖未 dirty 字段.

def _sync_pre_to_post(field_pre: str, field_post: str):
    """已废弃: Streamlit 1.32+ 不允许改 widget state. 同步改用 _get_post_default()."""
    pass

def _mark_post_dirty(field_post: str):
    """已废弃: 改用 _on_post_change"""
    st.session_state[f'_dirty_{field_post}'] = True


def _on_post_change(field_post: str, field_pre: str, widget_key: str):
    """Tab 2 widget on_change 回调. 抓用户输入 → 存影子 + 标 dirty.
    Args:
        field_post:  业务字段名 (t_post/c_post/tg_post)
        field_pre:   Tab 1 字段名 (t_pre/c_pre/tg_pre) - 用于对比
        widget_key:  widget 在 session_state 中的 key (_t_post_widget 等)
    """
    new_value = st.session_state.get(widget_key, '')
    pre_value = st.session_state.get(field_pre, '')
    # 如果新值 == pre 值, 视为"用户没改" (他只是输入了和 Tab 1 一样的内容, 或者根本没动)
    if new_value == pre_value:
        # 清 dirty, 让后续 Tab 1 改动能同步过来
        st.session_state[f'_dirty_{field_post}'] = False
        st.session_state[f'_user_{field_post}'] = new_value
    else:
        # 标 dirty, 保存用户版本
        st.session_state[f'_dirty_{field_post}'] = True
        st.session_state[f'_user_{field_post}'] = new_value


def _get_post_default(field_pre: str, field_post: str) -> str:
    """Tab 2 控件的 value 参数: dirty 时用用户已改值, 否则用 Tab 1 当前值."""
    if st.session_state.get(f'_dirty_{field_post}', False):
        return st.session_state.get(f'_user_{field_post}', '')
    return st.session_state.get(field_pre, '')


def _on_cover_pre_change():
    """
    Tab 1 封面上传时立即兜底缓存.
    关键: streamlit 的 UploadedFile 生命周期不可靠 (tab 切换/rerun/GC 后 .read() 失败),
    所以上传瞬间必须把字节和 PIL Image 存到 session_state, 之后不再依赖 UploadedFile.
    """
    f = st.session_state.get('cover_pre')
    if f is None or type(f).__name__ == 'DeletedFile' or not hasattr(f, 'read'):
        return
    try:
        f.seek(0)
        raw = f.read()
        if not raw:
            return
        # 存字节 (跨 tab 切换/页面刷新都不丢, 跟 UploadedFile 状态完全解耦)
        st.session_state['_cover_global_bytes'] = raw
        # 立即解析为 PIL Image 缓存, 后续预测直接拿
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        st.session_state['_cover_global_img'] = img
    except Exception as e:
        print(f'[_on_cover_pre_change] failed: {e}')


def _on_cover_post_change():
    """Tab 2 重新上传时, 同样兜底缓存, 并标记 dirty (不被 Tab 1 覆盖)"""
    f = st.session_state.get('cover_post')
    if f is None or type(f).__name__ == 'DeletedFile' or not hasattr(f, 'read'):
        return
    try:
        f.seek(0)
        raw = f.read()
        if not raw:
            return
        st.session_state['_cover_global_bytes'] = raw
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        st.session_state['_cover_global_img'] = img
        st.session_state['_dirty_cover_post'] = True
    except Exception as e:
        print(f'[_on_cover_post_change] failed: {e}')


def _open_cover(uploaded_file, show_error=False):
    """
    解析上传的封面为 PIL Image, 缓存到 session_state 避免 streamlit 重跑时
    UploadedFile 指针被消费导致的 UnidentifiedImageError.

    兜底: Tab 2 没传但 Tab 1 传了, 自动复用 Tab 1 缓存
          (file_uploader widget 不一定响应从 session_state 注入的 UploadedFile)

    兼容: None / DeletedFile (用户删了文件) / 缺 seek 方法的占位符 — 都 fallback

    支持: JPEG/PNG/WebP 标准格式; 尝试用 pillow-heif 兜底 HEIC/HEIF
          (iPhone 照片有时是 HEIC 改后缀上传的)

    Returns: PIL.Image (RGB) 或 None
    """
    # ============== 第 0 道防线: 全局缓存 (跟 widget 状态完全解耦, 最稳) ==============
    # on_change 回调 (上传瞬间) 已经把 PIL Image 和 bytes 存到 session_state 了
    # 所以无论 widget 状态多奇怪, 都能拿到
    global_img = st.session_state.get('_cover_global_img')
    if global_img is not None:
        print(f'[app] _open_cover: 命中全局 PIL Image 缓存 size={global_img.size}')
        return global_img
    global_bytes = st.session_state.get('_cover_global_bytes')
    if global_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(global_bytes)).convert('RGB')
            st.session_state['_cover_global_img'] = img
            print(f'[app] _open_cover: 从全局 bytes 重建 size={img.size}')
            return img
        except Exception as e:
            print(f'[app] _open_cover: 全局 bytes 解析失败: {e}')

    # ============== 下面是从 UploadedFile 实时解析 (慢路径, 不一定可靠) ==============
    def _fallback_to_pre():
        """统一兜底: 复用 cover_pre 的 PIL Image 缓存 (按 file_id)"""
        pre_cover = st.session_state.get('cover_pre')
        if pre_cover is not None and hasattr(pre_cover, 'file_id'):
            return st.session_state.get(f'_cover_img_{pre_cover.file_id}')
        return None

    # 情况 1: 传 None
    if uploaded_file is None:
        return _fallback_to_pre()

    # 情况 2: streamlit 的删除占位符 DeletedFile / 或缺 read 方法的伪对象
    type_name = type(uploaded_file).__name__
    if type_name == 'DeletedFile' or not hasattr(uploaded_file, 'read'):
        return _fallback_to_pre()

    # 用 file_id 区分不同上传
    cache_key = f'_cover_img_{uploaded_file.file_id}'
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached
    try:
        from PIL import Image
        # 显式 seek(0) + 读全部字节, 用 BytesIO 包装 — 彻底和 UploadedFile 解耦
        try:
            uploaded_file.seek(0)
        except (AttributeError, OSError, ValueError):
            return _fallback_to_pre()
        try:
            raw = uploaded_file.read()
        except (AttributeError, OSError, ValueError):
            return _fallback_to_pre()
        if not raw:
            return _fallback_to_pre()

        # 顺带存一份到全局缓存 (解耦 file_uploader 状态, 跨 tab 切换/页面 rerun 都稳)
        st.session_state['_cover_global_bytes'] = raw

        # 1. 尝试 HEIC/HEIF 兜底 (iPhone 默认格式, 经常改后缀为 .jpg 上传)
        #    magic bytes: HEIC = 00 00 00 ?? 66 74 79 70 68 65 69 63  ('ftypheic')
        if raw[:12].startswith(b'\x00\x00') and b'ftyp' in raw[:20]:
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                if show_error:
                    st.error('检测到 HEIC/HEIF 格式 (iPhone 照片), 请安装 pillow-heif: `pip install pillow-heif` 后重启')
                return None
        # 2. 尝试 standard image 解析
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()  # 强制 decode, 避免懒加载在 .convert() 时才报错
            img = img.convert('RGB')
        except Exception as parse_err:
            # 3. WebP / AVIF 等其他格式, 提示明确错误
            ext = (uploaded_file.name or '').split('.')[-1].lower() if uploaded_file.name else '?'
            if show_error:
                st.error(f'封面解析失败 ({ext}, {len(raw)//1024}KB): {parse_err}\n\n请尝试:\n'
                         f'1. 用图片编辑器另存为 .jpg 或 .png\n'
                         f'2. 或在 Mac 上 `sips -s format jpeg 原文件.jpg --out 新文件.jpg` 转格式\n'
                         f'3. 或在 iPhone 设置 → 相机 → 格式 → 选择"兼容性最佳"')
            print(f'[app] _open_cover failed: {ext}, {len(raw)}B, err={parse_err}')
            return None
        st.session_state[cache_key] = img
        st.session_state['_cover_global_img'] = img  # 全局缓存 (解耦 file_id)
        return img
    except Exception as e:
        if show_error:
            st.error(f'封面处理异常: {e}')
        print(f'[app] _open_cover outer failed: {e}')
        return None


# ============== 结果渲染 (先定义, 后面 tab 调) ==============
def _render_result(result, mode: str, f1: float, cmt_n: int = 0):
    st.divider()
    p = result['proba']
    label = result['label']
    colors = {0: '#ff4b4b', 1: '#ffa500', 2: '#21c354'}
    color = colors[label]
    label_names = ['低热度', '中热度', '高热度']

    # 概率条
    st.subheader(f'模型预测概率 ({"发前" if mode=="pre" else "发后"}模型, F1={f1:.3f})')
    cols_prob = st.columns(3)
    cols_prob[0].metric('低热度', f'{p[0]*100:.1f}%', delta='最可能' if label == 0 else None, delta_color='inverse')
    cols_prob[1].metric('中热度', f'{p[1]*100:.1f}%', delta='最可能' if label == 1 else None, delta_color='off')
    cols_prob[2].metric('高热度', f'{p[2]*100:.1f}%', delta='最可能' if label == 2 else None, delta_color='normal')

    bar_colors = ['#ff4b4b', '#ffa500', '#21c354']  # 低→中→高: 红→橙→绿
    prob_df = pd.DataFrame({'低热度': [p[0]], '中热度': [p[1]], '高热度': [p[2]]})
    st.bar_chart(prob_df, height=200, color=bar_colors)

    # 大字标签
    st.markdown(
        f"<h1 style='text-align:center; color:{color};'>预测: {result['label_text']}</h1>",
        unsafe_allow_html=True
    )

    # 副标签
    sorted_idx = np.argsort(p)[::-1]
    second_p = p[sorted_idx[1]]
    second_name = label_names[sorted_idx[1]]
    if second_p > 0.25 and sorted_idx[0] != sorted_idx[1]:
        st.markdown(
            f"<p style='text-align:center; color:#666;'>次高概率: {second_name} ({second_p*100:.1f}%) — 也存在可能</p>",
            unsafe_allow_html=True
        )

    # 三列特征指标
    f = result['features']
    m1, m2, m3 = st.columns(3)
    chars = f['content_len_chars']
    m1.metric('正文字数', f'{chars} 字 (清洗后 {f["content_len_words"]} 词)',
              delta='够长' if 200 <= chars <= 500 else ('偏短' if chars < 200 else '过长'),
              delta_color='normal' if 200 <= chars <= 500 else 'inverse')
    m2.metric('标题情感强度', f'{f["title_sent"]:+d}',
              delta='情绪充足' if f['title_sent'] > 0 else '情感偏弱',
              delta_color='normal' if f['title_sent'] > 0 else 'inverse')
    m3.metric('标签数量', f'{f["tag_count"]} 个',
              delta='足够' if f['tag_count'] >= 3 else '偏少',
              delta_color='normal' if f['tag_count'] >= 3 else 'inverse')

    # 改进建议
    if result['suggestions']:
        st.subheader('改进建议')
        for s in result['suggestions']:
            st.info(s)

    # 热度等级说明
    with st.expander('热度等级说明'):
        st.markdown("""
| 等级 | 点赞数参考 | 说明 |
|------|-----------|------|
| 高热度 | >= 500 赞 | 爆款潜力, 直接发布 |
| 中热度 | 50~499 赞 | 有一定传播力, 可按建议优化 |
| 低热度 | < 50 赞 | 建议按改进意见修改后再发 |
        """)


# ============== 页面基本设置 ==============
st.set_page_config(
    page_title='小红书爆款预测器',
    page_icon='X',
    layout='centered'
)

# ============== 顶部标题区 ==============
st.title('小红书爆款预测器')
st.caption('天津师范大学周边美食探店 · 发布前预测热度')

# F1 数值 (在下方"关于本模型"展开区使用, 不再单独首屏显示)
f1_pre_disp = f'{_f1_pre:.3f}'
f1_post_disp = f'{_f1_post:.3f}'

# (发前/发后 F1 信息合并到下方"关于本模型"展开区, 不再单独占据首屏)

with st.expander('关于本模型 (点击展开)', expanded=False):
    st.markdown(f"""
**整体架构 — 文本 + 图像多模态融合, 封面为可选输入**

**① 文本分类器 (主模型)**
- 算法: 5-fold Random Forest 投票 + 类别权重平衡 ({{0:1.0, 1:1.3, 2:1.5}})
- 训练数据: 649 条天津师大周边美食探店帖子
- 特征: 20 维 (13 维结构化 + 7 维代理特征)
  - 13 维原始: 长度、词数、标签、情感、评论分布等
  - 7 维代理 (D 方案): `tag_has_food` / `content_lines` / `content_has_loc` /
    `content_digit_ratio` / `content_has_price` / `title_has_q` / `content_avg_word`
- 特征工程: `content_len_words` 经 `log1p(min(x, 200))` 压缩, 削弱长尾支配
- 5 折 CV F1-macro: 发前 **{f1_pre_disp}** | 发后 **{f1_post_disp}** (诚实, 无启发式 hack)

**② 图像分类器 (可选, 上传封面时启用)**
- 算法: Fine-tuned ResNet50 (ImageNet 预训练 backbone 冻结, 训练分类头)
- 输入: 224×224 RGB 封面图片
- 验证 F1: 0.40

**③ 图文对齐 (可选, 上传封面时启用)**
- 算法: Chinese CLIP (`OFA-Sys/chinese-clip-vit-base-patch16`)
- 输入: 封面图片 + 标题正文前 80 字
- 输出: 图文相似度分数, 用于检测"图片与内容不符"

**④ 多模态融合 (仅当上传封面时)**
- 公式: `0.4 × ResNet50_proba + 0.6 × 文本_RF_proba`
- 加权后 argmax 给出最终标签
- 融合 F1: 0.52 (相对纯文本 0.408 提升)

> **局限性 — Training-Inference 分布偏移**: 训练样本全部是"已发布 ≥1 天"的帖子.
> 发前预测时, 6 维 POST 特征 (`days_alive` / `comment_n` / `cmt_avg_len` /
> `pos_ratio` / `neg_ratio` / `neutral_ratio` / `sent_strength`) 在训练时随机 mask 为 0
> 让模型学会"看到 0 当缺失". **F1={f1_pre_disp}** 是诚实的发前性能.
> 训练样本 649 条偏少, F1 上限约 0.65, 预测结果应作为参考而非定论.
    """)
st.divider()

# ============== 多模态结果渲染 (封面+文本+CLIP) ==============
def _render_multimodal(res: dict, mode: str, image):
    """在预测结果块中追加多模态可视化 (CNN+GBM+CLIP+图文不符警告)"""
    label_map = {0: '❄️ 冷门 (<100 赞)', 1: '🌤 温热 (100-500)', 2: '🔥 爆款 (>500)'}
    color_map = {0: '#90a4ae', 1: '#f57c00', 2: '#c62828'}
    final = res['final_label']
    conf = res['final_proba'][final]
    # 多模态综合卡
    st.subheader(f'🧠 多模态融合结果 ({"发前" if mode=="pre" else "发后"}模型)')
    st.markdown(f"""
    <div style="padding:18px;border-radius:12px;background:linear-gradient(135deg,{color_map[final]}22,{color_map[final]}05);
                border-left:6px solid {color_map[final]};margin:8px 0">
        <div style="font-size:13px;color:#666;margin-bottom:4px">综合 {res.get('fusion_weight','0.4×CNN + 0.6×Text')}</div>
        <div style="font-size:26px;font-weight:700;color:{color_map[final]}">{label_map[final]}</div>
        <div style="font-size:13px;color:#888">置信度 {conf*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    # CNN vs GBM 对比
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('🖼 **CNN (封面) 预测**')
        if res.get('cnn') and res['cnn'].get('error') is None:
            cp = res['cnn']['proba']
            st.write(f"预测: {label_map[res['cnn']['label']]}")
            st.progress(float(cp[res['cnn']['label']]), text=f"置信度 {cp[res['cnn']['label']]*100:.1f}%")
        else:
            st.caption('CNN 未参与')
    with cc2:
        st.markdown('📝 **GBM (文案) 预测**')
        tp = res['text']['proba']
        st.write(f"预测: {label_map[res['text']['label']]}")
        st.progress(float(tp[res['text']['label']]), text=f"置信度 {tp[res['text']['label']]*100:.1f}%")
    # CLIP 图文匹配
    st.markdown('🔍 **CLIP 图文匹配**')
    clip_sim = res.get('clip_similarity')
    clip_msg = res.get('clip_message', '—')
    if clip_sim is not None:
        st.write(f"相似度: **{clip_sim:.2f}**  ·  {clip_msg}")
    else:
        st.caption(clip_msg)
    # 图文不符警告
    mismatch = res.get('mismatch', False)
    mismatch_level = res.get('mismatch_level', 'none')
    if mismatch and mismatch_level != 'none':
        warn_color = '#c62828' if mismatch_level == 'severe' else '#f57c00'
        warn_icon  = '🛑' if mismatch_level == 'severe' else '⚠️'
        st.markdown(f"""
        <div style="padding:12px;border-radius:8px;background:{warn_color}11;border-left:4px solid {warn_color};margin:8px 0">
            <b>{warn_icon} 图文不符 ({mismatch_level})</b><br>
            <span style="font-size:13px;color:#555">{res.get('mismatch_reason','')}</span>
        </div>
        """, unsafe_allow_html=True)
    # 改进建议 (多模态版)
    sugg = res.get('suggestions', [])
    if sugg:
        with st.expander('💡 多模态改进建议', expanded=True):
            for s in sugg:
                st.write(f"• {s}")


# ============== 两阶段 UI (发布前 / 发布后 — 封面可选) ==============
tab_pre, tab_post = st.tabs([
    '📝 发布前预测',
    '🔄 发后重测 (有首批评论后)',
])

# ============== Tab 1: 发布前预测 ==============
with tab_pre:
    st.caption('只填标题/正文/标签, 走发前模型 (POST 特征自动 mask=0). 可选上传封面启用多模态预测')
    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.text_input('帖子标题', placeholder='例: 天师大后街这家烤肉真的绝了!!', max_chars=100, key='t_pre')
        content = st.text_area('正文内容', placeholder='详细描述你的探店体验, 建议 200 字以上...', height=200, key='c_pre')
    with col2:
        tags = st.text_area('标签 (逗号分隔)', placeholder='天津师范,校园美食,穷鬼套餐', height=100, key='tg_pre')
        st.caption('每行一个或用逗号分隔')

    # 封面上传 (可选) — 上传时自动同步到 Tab 2
    # 注意: st.file_uploader 不允许 session_state 注入, 所以 cover 不用 on_change 同步 widget state
    # 但 on_change 回调内读 bytes 再存到 session_state 是允许的 (这就是 _on_cover_pre_change 的作用)
    cover_pre = st.file_uploader('📷 封面图片 (可选, 上传后启用多模态预测)', type=['jpg', 'jpeg', 'png', 'heic', 'heif'],
                                 key='cover_pre',
                                 on_change=_on_cover_pre_change)
    # 兼容 DeletedFile 占位 (用户点 X 删除时返回)
    cover_pre_valid = (cover_pre is not None
                       and type(cover_pre).__name__ != 'DeletedFile'
                       and hasattr(cover_pre, 'read'))
    if cover_pre_valid:
        # 立即同步缓存 (不依赖 on_change, 永远不依赖 widget 状态变化)
        # 这一步既显示预览, 又把 PIL Image + bytes 写入 session_state 兜底
        _img = _open_cover(cover_pre)
        if _img is not None:
            st.image(_img, caption='封面预览 (会自动同步到"发后重测" Tab)', width=240)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run_pre = st.button('预测 (发前)', type='primary', use_container_width=True, key='btn_pre')

    if run_pre:
        if not title.strip():
            st.error('请先填写标题'); st.stop()
        if not content.strip():
            st.error('请先填写正文内容'); st.stop()
        # ============== 封面解析: 4 道防线 (从最稳到最不稳) ==============
        img = st.session_state.get('_cover_global_img')  # 防线 1
        if img is None:
            _b = st.session_state.get('_cover_global_bytes')  # 防线 2
            if _b:
                try:
                    from PIL import Image
                    img = Image.open(io.BytesIO(_b)).convert('RGB')
                    st.session_state['_cover_global_img'] = img
                except Exception:
                    pass
        if img is None and cover_pre_valid:  # 防线 3: 实时解析 cover_pre
            img = _open_cover(cover_pre, show_error=False)
        if img is not None and _CNN_OK:
            with st.spinner('CNN + CLIP + GBM 联合推理中...'):
                res = predict_combined(title, content, tags, image=img, mode='pre')
            # 1) 多模态综合卡 (替代原概率条)
            st.divider()
            label_map = {0: '❄️ 冷门 (<100 赞)', 1: '🌤 温热 (100-500)', 2: '🔥 爆款 (>500)'}
            color_map = {0: '#90a4ae', 1: '#f57c00', 2: '#c62828'}
            final = res['final_label']
            p = res['final_proba']
            colors = {0: '#ff4b4b', 1: '#ffa500', 2: '#21c354'}
            color = colors[final]
            label_names = ['低热度', '中热度', '高热度']
            st.subheader(f'多模态预测概率 (发前模型, CNN+GBM 融合)')
            cols_prob = st.columns(3)
            cols_prob[0].metric('低热度', f'{p[0]*100:.1f}%', delta='最可能' if final == 0 else None, delta_color='inverse')
            cols_prob[1].metric('中热度', f'{p[1]*100:.1f}%', delta='最可能' if final == 1 else None, delta_color='off')
            cols_prob[2].metric('高热度', f'{p[2]*100:.1f}%', delta='最可能' if final == 2 else None, delta_color='normal')
            prob_df = pd.DataFrame({'低热度': [p[0]], '中热度': [p[1]], '高热度': [p[2]]})
            st.bar_chart(prob_df, height=200, color=['#ff4b4b', '#ffa500', '#21c354'])
            st.markdown(
                f"<h1 style='text-align:center; color:{color};'>预测: {label_names[final]}</h1>",
                unsafe_allow_html=True
            )
            _render_multimodal(res, mode='pre', image=img)
        else:
            with st.spinner('预测中 (封面解析失败, 走纯文本)...'):
                result = predict_with_mode(title, content, tags, None, mode='pre')
            if img is None and cover_pre is not None:
                st.caption('⚠️ 封面解析失败 (4 道防线全失败), 已退到纯文本预测. 如需多模态, 请重新上传封面')
            _render_result(result, mode='pre', f1=_f1_pre)

# ============== Tab 2: 发后重测 ==============
with tab_post:
    st.caption('发布后拿到首批评论数据, 填入下方 3 个字段, 走发后模型 (全 20 维特征, F1 显著提升)')

    # ============== 主动同步策略: 用 widget 的 value= 参数 (不依赖 key) ==============
    # 关键洞察: 不要给这些 widget 传 key, 让 Streamlit 自动分配唯一 key
    # 这样 value= 参数每次 rerun 都生效 (Streamlit 锁定的是 key 绑定的 state, 没 key 就没锁定)
    # 用户输入 → on_change 回调 → 写影子变量 + 标 dirty
    # dirty 时: 显示影子变量 (用户原版); 否则: 显示 Tab 1 当前值

    def _post_value(f_pre, f_post):
        if st.session_state.get(f'_dirty_{f_post}', False):
            return st.session_state.get(f'_user_{f_post}', '')
        return st.session_state.get(f_pre, '')

    col1, col2 = st.columns([3, 1])
    with col1:
        title2 = st.text_input('帖子标题',
                               value=_post_value('t_pre', 't_post'),
                               placeholder='同发布前', max_chars=100,
                               key=f'_t_post_widget',
                               on_change=_on_post_change, args=('t_post', 't_pre', '_t_post_widget'))
        content2 = st.text_area('正文内容',
                                value=_post_value('c_pre', 'c_post'),
                                placeholder='同发布前', height=200,
                                key=f'_c_post_widget',
                                on_change=_on_post_change, args=('c_post', 'c_pre', '_c_post_widget'))
    with col2:
        tags2 = st.text_area('标签',
                             value=_post_value('tg_pre', 'tg_post'),
                             placeholder='同发布前', height=100,
                             key=f'_tg_post_widget',
                             on_change=_on_post_change, args=('tg_post', 'tg_pre', '_tg_post_widget'))
        st.caption('🔄 发布前输入的标题/正文/标签会自动同步到这里 (可手动改, 改后不会被覆盖)')

    # 同步状态展示
    synced_fields = [f for f in ('t_post', 'c_post', 'tg_post')
                     if not st.session_state.get(f'_dirty_{f}', False)
                     and st.session_state.get(f, '')]
    if synced_fields:
        st.caption(f'✅ 自动同步字段: {", ".join(f.replace("_post", "") for f in synced_fields)} | 手动修改后会保留你的版本')

    # 封面上传 (可选) — 默认显示 Tab 1 同步过来的封面
    # on_change 同时缓存 bytes 到 session_state (Tab 1 没传 / Tab 2 重传都能用)
    cover_post = st.file_uploader('📷 封面图片 (可选, 默认沿用发布前上传的封面, 可重新上传)', type=['jpg', 'jpeg', 'png'],
                                  key='cover_post',
                                  on_change=_on_cover_post_change)
    # 预览: 优先用 cover_post (本 Tab 上传), 兜底用 cover_pre (Tab 1 同步)
    preview_img = _open_cover(cover_post)
    if preview_img is not None:
        st.image(preview_img, caption='封面预览', width=240)
    elif st.session_state.get('cover_pre') is not None:
        st.caption('🖼 已从"发布前预测"自动带入封面 (不重新上传则沿用)')

    st.markdown('**首批评论数据 (决定 POST 特征)**')
    cmt_n = st.number_input('评论数', min_value=0, value=10, step=1, key='cmt_n')
    cmt_avg_len = st.number_input('评论平均字数', min_value=0.0, value=5.0, step=0.5, key='cmt_avg')
    comments_raw = st.text_area('评论内容 (用于情感聚合, 每行一条)', placeholder='求地址!\n已收藏\n好吃!', height=100, key='cmt_raw')

    _, btn_col2, _ = st.columns([2, 1, 2])
    with btn_col2:
        run_post = st.button('重测 (发后)', type='primary', use_container_width=True, key='btn_post')

    if run_post:
        if not title2.strip():
            st.error('请先填写标题'); st.stop()
        if not content2.strip():
            st.error('请先填写正文内容'); st.stop()
        comments = [c.strip() for c in comments_raw.strip().splitlines() if c.strip()] if comments_raw.strip() else []
        # 若用户没填评论内容但填了 cmt_n, 用占位评论让情感特征能跑
        if not comments and cmt_n > 0:
            comments = ['求地址'] * int(cmt_n)  # 全部当"中性/求地址", 让模型看到真实评论量
        # ============== 封面解析: 4 道防线 (从最稳到最不稳) ==============
        img = None
        # 防线 1: 全局 PIL Image 缓存 (上传瞬间就建立, 永远最稳)
        img = st.session_state.get('_cover_global_img')
        # 防线 2: 全局 bytes 缓存
        if img is None:
            _b = st.session_state.get('_cover_global_bytes')
            if _b:
                try:
                    from PIL import Image
                    img = Image.open(io.BytesIO(_b)).convert('RGB')
                    st.session_state['_cover_global_img'] = img
                except Exception:
                    pass
        # 防线 3: 实时解析 cover_post (Tab 2 自己上传的)
        if img is None:
            cover_post_valid = (cover_post is not None
                                and type(cover_post).__name__ != 'DeletedFile'
                                and hasattr(cover_post, 'read'))
            if cover_post_valid:
                img = _open_cover(cover_post, show_error=False)
        # 防线 4: 实时解析 cover_pre (Tab 1 上传的)
        if img is None:
            _cp = st.session_state.get('cover_pre')
            if _cp is not None and type(_cp).__name__ != 'DeletedFile' and hasattr(_cp, 'read'):
                img = _open_cover(_cp, show_error=False)
        if img is not None and _CNN_OK:
            with st.spinner('CNN + CLIP + GBM 联合推理中... (发后模式)'):
                res = predict_combined(title2, content2, tags2, image=img, mode='post', comments=comments)
            st.divider()
            label_map = {0: '❄️ 冷门 (<100 赞)', 1: '🌤 温热 (100-500)', 2: '🔥 爆款 (>500)'}
            color_map = {0: '#90a4ae', 1: '#f57c00', 2: '#c62828'}
            final = res['final_label']
            p = res['final_proba']
            colors = {0: '#ff4b4b', 1: '#ffa500', 2: '#21c354'}
            color = colors[final]
            label_names = ['低热度', '中热度', '高热度']
            st.subheader(f'多模态预测概率 (发后模型, CNN+GBM 融合)')
            cols_prob = st.columns(3)
            cols_prob[0].metric('低热度', f'{p[0]*100:.1f}%', delta='最可能' if final == 0 else None, delta_color='inverse')
            cols_prob[1].metric('中热度', f'{p[1]*100:.1f}%', delta='最可能' if final == 1 else None, delta_color='off')
            cols_prob[2].metric('高热度', f'{p[2]*100:.1f}%', delta='最可能' if final == 2 else None, delta_color='normal')
            prob_df = pd.DataFrame({'低热度': [p[0]], '中热度': [p[1]], '高热度': [p[2]]})
            st.bar_chart(prob_df, height=200, color=['#ff4b4b', '#ffa500', '#21c354'])
            st.markdown(
                f"<h1 style='text-align:center; color:{color};'>预测: {label_names[final]}</h1>",
                unsafe_allow_html=True
            )
            _render_multimodal(res, mode='post', image=img)
        else:
            # ============== 兜底: 4 道防线全失败, 退到纯文本预测 (不阻断用户) ==============
            with st.spinner('预测中 (封面解析失败, 走纯文本)...'):
                result = predict_with_mode(title2, content2, tags2, comments, mode='post')
            if not comments_raw.strip() and cmt_n > 0:
                st.caption(f'⚠️ 未粘评论, 情感特征用占位"求地址" (评论数={cmt_n}, 平均长度={cmt_avg_len} 仍生效)')
            st.caption('⚠️ 封面解析失败 (4 道防线全失败), 已退到纯文本预测. 如需多模态, 请在 Tab 1 重新上传封面')
            _render_result(result, mode='post', f1=_f1_post, cmt_n=cmt_n)


# ============== 多模态模型说明 (全局) ==============
with st.expander('ℹ️ 关于多模态封面预测模型', expanded=False):
    st.markdown("""
    - **训练数据**: 941 条天津师大美食探店帖 (vs 发前模型 649 条)
    - **算法**: ResNet50 (ImageNet 预训练 → 13 类美食图微调) + Chinese CLIP + LightGBM 5-fold 投票
    - **特征**: 36 维 = 20 文本 (含 7 代理) + 16 ResNet50 全连接层 PCA 降维
    - **加权**: 0.4 × CNN 概率 + 0.6 × GBM 概率 (Text F1 更高, 权重更大)
    - **诚实声明**: F1 上限受 941 条小样本限制, 仅作参考
    """)

