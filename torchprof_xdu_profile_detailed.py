import functools
from collections import defaultdict, namedtuple
import torch
from torch.profiler import profile, ProfilerActivity

Trace = namedtuple("Trace", ["path", "leaf", "module", "module_name"])


def walk_modules(module, name="", path=()):
    """递归遍历模型所有模块，输出 Trace(path, leaf, module, module_name)"""
    if not name:
        module_name = module.__class__.__name__
    else:
        module_name = name

    named_children = list(module.named_children())
    current_path = path + (module_name,)
    yield Trace(current_path, len(named_children) == 0, module, module_name)
    for child_name, child_module in named_children:
        yield from walk_modules(child_module, name=child_name, path=current_path)


class ProfileDetailed(object):
    """
    torchprof-xdu: 用 torch.profiler 实现的逐层 profile，兼容原 torchprof 接口。
    支持 with 语法、paths 精选、raw、display、自动递归注册/恢复 forward。
    新增记录 FLOPs 和更详细的内存信息。
    支持排序和 top-k 显示。
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
        self.trace_module_details = defaultdict(lambda: {"params": 0, "flops": 0})

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof-xdu profiler is not reentrant")
        self.entered = True
        self._forwards = {}
        current_traces = []
        for trace in walk_modules(self._model):
            current_traces.append(trace)
            self._hook_trace(trace)
            params = sum(p.numel() for p in trace.module.parameters() if p.requires_grad)
            self.trace_module_details[trace.path]["params"] = params
        self.traces = tuple(current_traces)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        for trace in self.traces:
            self._remove_hook_trace(trace)
        if hasattr(self, '_forwards'): # Ensure _forwards exists before deleting
            del self._forwards
        self.exited = True

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module, module_name] = trace
        _id = id(module)
        if (self.paths is not None and path in self.paths) or \
           (self.paths is None and leaf):
            if _id in self._ids:
                return
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                activities = [ProfilerActivity.CPU]
                if self.use_cuda:
                    activities.append(ProfilerActivity.CUDA)
                
                profiler_kwargs = {
                    "activities": activities,
                    "record_shapes": True,
                    "profile_memory": self.profile_memory,
                }
                if torch.__version__ >= "1.8": # hasattr(torch.profiler, "with_flops") is less reliable
                    profiler_kwargs["with_flops"] = True

                with profile(**profiler_kwargs) as prof:
                    res = _forward(*args, **kwargs)
                
                events = prof.key_averages()
                self.trace_profile_events[path].append(events)
                return res

            module.forward = wrap_forward
        # Removed "return trace" as it's not used by the caller loop in __enter__

    def _remove_hook_trace(self, trace):
        [path, leaf, module, module_name] = trace
        _id = id(module)
        if _id in self._ids:
            if path in self._forwards:
                module.forward = self._forwards[path]
                del self._forwards[path]
            self._ids.discard(_id)

    def raw(self):
        """返回 (traces, trace_profile_events, trace_module_details)，可用于自定义 display"""
        if self.exited:
            return (self.traces, self.trace_profile_events, self.trace_module_details)
        else:
            return None

    def display(self, show_events=False, sort_by=None, top_k=-1):
        """
        树状结构表格输出。
        新增排序和 top-k 功能。

        Args:
            show_events (bool): 是否显示更详细的 individual profiler events。
            sort_by (str, optional): 用于排序的指标名称 (例如 "FLOPs", "Self CPU total")。
                                     默认为 None (不排序)。
            top_k (int, optional): 显示前 k 个结果。默认为 -1 (显示全部)。
        """
        if self.exited:
            from .torchprof_xdu_display_detailed import traces_to_display_detailed

            return traces_to_display_detailed(
                traces=self.traces,
                trace_profile_events=self.trace_profile_events,
                trace_module_details=self.trace_module_details,
                show_events=show_events,
                paths=self.paths,
                use_cuda=self.use_cuda,
                profile_memory=self.profile_memory,
                sort_by=sort_by,
                top_k=top_k
            )
        return "<unfinished torchprof-xdu.profile_detailed>"