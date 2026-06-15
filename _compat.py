"""
加载 V3 模型时的 numpy 兼容性补丁
V3 模型在 numpy 2.x 下训练保存, 加载到 numpy 1.x 环境时需要打多个补丁:
  1. numpy._core.numeric -> numpy.core.numeric (numpy 2.x 内部模块重命名)
  2. numpy._core.* (其他子模块同理)
  3. numpy.random._pickle.MT19937 反序列化 (numpy 1.x pickle 传 class, 2.x 传 string)
  4. numpy 1.x 的 RandomState() 只接受 int seed, 不接受 BitGenerator 实例 (2.x 才支持)
     需要把 __randomstate_ctor 改成: 创建空 RandomState + set_state(legacy state)
"""
import joblib
import numpy.core as _np_core


def _patch_numpy_modules():
    """把 numpy._core.* 全映射到 numpy.core.* (numpy 2.x -> 1.x)"""
    import sys
    if 'numpy._core' in sys.modules and not hasattr(sys.modules['numpy._core'], '_patched_marker'):
        return  # 已 patch
    _core = type(sys)('numpy._core')
    _core._patched_marker = True
    # 把 numpy.core.* 的所有子模块复制到 numpy._core
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('numpy.core.'):
            sub = mod_name.split('numpy.core.', 1)[1]
            sys.modules[f'numpy._core.{sub}'] = sys.modules[mod_name]
    # 直接代理 numpy.core 下的属性
    _core.numeric = _np_core.numeric
    _core.multiarray = _np_core.multiarray
    _core.umath = _np_core.umath
    _core._multiarray_umath = getattr(_np_core, '_multiarray_umath', None)
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.numeric'] = _np_core.numeric
    sys.modules['numpy._core.multiarray'] = _np_core.multiarray
    sys.modules['numpy._core.umath'] = _np_core.umath


def _patch_numpy_pickle():
    """重写 __bit_generator_ctor 和 __randomstate_ctor, 支持 numpy 2.x pickle 格式"""
    import numpy.random._pickle as _pickle
    _orig_bitgen = _pickle.__bit_generator_ctor
    _BitGenerators = _pickle.BitGenerators

    def _patched_bitgen(bit_generator_name='MT19937'):
        if isinstance(bit_generator_name, type):
            name = bit_generator_name.__name__
            if name in _BitGenerators:
                return _BitGenerators[name]()
            return bit_generator_name()
        return _orig_bitgen(bit_generator_name)

    def _patched_randomstate(bit_generator_name='MT19937',
                              bit_generator_ctor=_patched_bitgen):
        """
        numpy 1.x 的 RandomState(seed) 只接受 int seed, 不接受 BitGenerator 实例.
        numpy 2.x 改为 RandomState(bit_generator_instance).
        解决: 创建一个空 RandomState, 然后把 state 注入进去.
        退而求其次: 如果 set_state 因 1.x/2.x state 格式差异失败, 用 fresh seed 即可
        (RandomState 只用于 train 时的 split, predict 阶段不影响).
        """
        from numpy.random import RandomState
        bg = bit_generator_ctor(bit_generator_name)
        rs = RandomState()
        try:
            rs.set_state(bg.state)
        except (ValueError, TypeError):
            # 1.x/2.x state 格式不兼容, 用 fresh state 即可
            # (V3 模型是分类器, 训练已结束, 推理阶段不依赖 random state)
            pass
        return rs

    _pickle.__bit_generator_ctor = _patched_bitgen
    _pickle.__randomstate_ctor = _patched_randomstate


def load_v3_compat(path: str):
    """加载 V3 模型, 自动处理 numpy 版本差异"""
    _patch_numpy_modules()
    _patch_numpy_pickle()
    return joblib.load(path)


if __name__ == '__main__':
    b = load_v3_compat('/Users/ming/Downloads/tare/Final/streamlit_app/models/best_classifier_pre_img.pkl')
    print('V3 loaded, n folds:', len(b.get('fold_models', [])))
