import functools
from collections import defaultdict, namedtuple
from torch.profiler import profile, ProfilerActivity

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def walk_modules(module, name="", path=()):
    """递归遍历模型所有模块，输出 Trace(path, leaf, module)"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0, module)
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class Profile(object):
    """
    torchprof-xdu: 用 torch.profiler 实现的逐层 profile，兼容原 torchprof 接口。
    支持 with 语法、paths 精选、raw、display、自动递归注册/恢复 forward。
    """

    def __init__(
        self, model, enabled=True, use_cuda=False, profile_memory=False, paths=None
    ):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof-xdu profiler is not reentrant")
        self.entered = True
        self._forwards = {}
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards
        self.exited = True

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        # 只对 paths 指定或叶子模块注册 profile
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            if _id in self._ids:
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                activities = [ProfilerActivity.CPU]
                if self.use_cuda:
                    activities.append(ProfilerActivity.CUDA)
                with profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=self.profile_memory,
                ) as prof:
                    res = _forward(*args, **kwargs)
                # 采集本层的所有事件
                self.trace_profile_events[path].append(prof.key_averages())
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            module.forward = self._forwards[path]

    def raw(self):
        """返回 (traces, trace_profile_events)，可用于自定义 display"""
        if self.exited:
            return (self.traces, self.trace_profile_events)

    def display(self, show_events=False):
        """树状结构表格输出，兼容 torchprof 的 display.py"""
        if self.exited:
            from .torchprof_xdu_display import traces_to_display

            return traces_to_display(
                self.traces,
                self.trace_profile_events,
                show_events=show_events,
                paths=self.paths,
                use_cuda=self.use_cuda,
                profile_memory=self.profile_memory,
            )
        return "<unfinished torchprof-xdu.profile>"
