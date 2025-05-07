import torch
from torch.profiler import profile, ProfilerActivity, record_function
from collections import OrderedDict, namedtuple

Measure = namedtuple(
    "Measure",
    [
        "self_cpu_total",
        "cpu_total",
        "self_cuda_total",
        "cuda_total",
        "self_cpu_memory",
        "cpu_memory",
        "self_cuda_memory",
        "cuda_memory",
        "occurrences",
    ],
)


def _format_time(t):
    if t is None:
        return ""
    if t >= 1e6:
        return f"{t/1e6:.3f}ms"
    elif t >= 1e3:
        return f"{t/1e3:.3f}us"
    elif t > 0:
        return f"{t:.3f}ns"
    else:
        return ""


def _format_memory(m):
    if m is None:
        return ""
    if m >= 1024 * 1024:
        return f"{m/1024/1024:.2f} Mb"
    elif m >= 1024:
        return f"{m/1024:.2f} Kb"
    elif m > 0:
        return f"{m:.0f} b"
    else:
        return ""


def _format_measure_tuple(measure):
    self_cpu_total = _format_time(measure.self_cpu_total) if measure else ""
    cpu_total = _format_time(measure.cpu_total) if measure else ""
    self_cuda_total = (
        _format_time(measure.self_cuda_total)
        if measure and measure.self_cuda_total is not None
        else ""
    )
    cuda_total = _format_time(measure.cuda_total) if measure else ""
    self_cpu_memory = (
        _format_memory(measure.self_cpu_memory)
        if measure and measure.self_cpu_memory is not None
        else ""
    )
    cpu_memory = (
        _format_memory(measure.cpu_memory)
        if measure and measure.cpu_memory is not None
        else ""
    )
    self_cuda_memory = (
        _format_memory(measure.self_cuda_memory)
        if measure and measure.self_cuda_memory is not None
        else ""
    )
    cuda_memory = (
        _format_memory(measure.cuda_memory)
        if measure and measure.cuda_memory is not None
        else ""
    )
    occurrences = str(measure.occurrences) if measure else ""
    return Measure(
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_cuda_total,
        cuda_total=cuda_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def _flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(_flatten_tree(st, depth=depth + 1))
    return flat


def _build_measure_tuple(events, occurrences):
    self_cpu_memory = None
    has_self_cpu_memory = any(hasattr(e, "self_cpu_memory_usage") for e in events)
    if has_self_cpu_memory:
        self_cpu_memory = sum([getattr(e, "self_cpu_memory_usage", 0) for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = sum([getattr(e, "cpu_memory_usage", 0) for e in events])
    self_cuda_memory = None
    has_self_cuda_memory = any(hasattr(e, "self_cuda_memory_usage") for e in events)
    if has_self_cuda_memory:
        self_cuda_memory = sum(
            [getattr(e, "self_cuda_memory_usage", 0) for e in events]
        )
    cuda_memory = None
    has_cuda_memory = any(hasattr(e, "cuda_memory_usage") for e in events)
    if has_cuda_memory:
        cuda_memory = sum([getattr(e, "cuda_memory_usage", 0) for e in events])
    self_cuda_total = None
    has_self_cuda_time = any(hasattr(e, "self_cuda_time_total") for e in events)
    if has_self_cuda_time:
        self_cuda_total = sum([getattr(e, "self_cuda_time_total", 0) for e in events])
    return Measure(
        self_cpu_total=sum([getattr(e, "self_cpu_time_total", 0) for e in events]),
        cpu_total=sum([getattr(e, "cpu_time_total", 0) for e in events]),
        self_cuda_total=self_cuda_total,
        cuda_total=sum([getattr(e, "cuda_time_total", 0) for e in events]),
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def group_by(events, keyfn):
    event_groups = OrderedDict()
    for event in events:
        key = keyfn(event)
        key_events = event_groups.get(key, [])
        key_events.append(event)
        event_groups[key] = key_events
    return event_groups.items()


def traces_to_display(
    traces,
    trace_profile_events,
    show_events=False,
    paths=None,
    use_cuda=False,
    profile_memory=False,
    dt=("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " "),  # 树状符号
):
    """
    完全仿照 torchprof 源码，支持 show_events、paths、use_cuda、profile_memory
    """
    tree = OrderedDict()
    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # 展开所有事件（支持多次 forward）
        events = [te for t_events in trace_profile_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and (
                (paths is None and leaf) or (paths is not None and path in paths)
            ):
                # 支持 show_events: 事件分组
                if show_events:
                    for event_name, event_group in group_by(events, lambda e: e.key):
                        event_group = list(event_group)
                        current_tree[name][event_name] = {
                            None: _build_measure_tuple(event_group, len(event_group))
                        }
                else:
                    current_tree[name][None] = _build_measure_tuple(
                        events, len(trace_profile_events[path])
                    )
            current_tree = current_tree[name]

    tree_lines = _flatten_tree(tree)

    # 动态列选择
    format_lines = []
    has_self_cuda_total = False
    has_self_cpu_memory = False
    has_cpu_memory = False
    has_self_cuda_memory = False
    has_cuda_memory = False

    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line
        # 树状符号
        next_depths = [pl[0] for pl in tree_lines[idx + 1 :]]
        pre = ""
        if depth > 0:
            pre = dt[1] if depth in next_depths and next_depths[0] >= depth else dt[2]
            depth -= 1
        while depth > 0:
            pre = (dt[0] + pre) if depth in next_depths else (dt[3] + pre)
            depth -= 1
        format_lines.append([pre + name, *_format_measure_tuple(measures)])
        if measures:
            has_self_cuda_total = has_self_cuda_total or (
                measures.self_cuda_total not in ("", None)
            )
            has_self_cpu_memory = has_self_cpu_memory or (
                measures.self_cpu_memory not in ("", None)
            )
            has_cpu_memory = has_cpu_memory or (measures.cpu_memory not in ("", None))
            has_self_cuda_memory = has_self_cuda_memory or (
                measures.self_cuda_memory not in ("", None)
            )
            has_cuda_memory = has_cuda_memory or (
                measures.cuda_memory not in ("", None)
            )

    heading = (
        "Module",
        "Self CPU total",
        "CPU total",
        "Self CUDA total",
        "CUDA total",
        "Self CPU Mem",
        "CPU Mem",
        "Self CUDA Mem",
        "CUDA Mem",
        "Number of Calls",
    )
    keep_indexes = [0, 1, 2, 9]
    if profile_memory:
        if has_self_cpu_memory:
            keep_indexes.append(5)
        if has_cpu_memory:
            keep_indexes.append(6)
    if use_cuda:
        if has_self_cuda_total:
            keep_indexes.append(3)
        keep_indexes.append(4)
        if profile_memory:
            if has_self_cuda_memory:
                keep_indexes.append(7)
            if has_cuda_memory:
                keep_indexes.append(8)
    keep_indexes = tuple(sorted(set(keep_indexes)))
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]
    display = (
        " | ".join(
            [
                "{:<{}s}".format(heading[keep_index], max_lens[keep_index])
                for keep_index in keep_indexes
            ]
        )
        + "\n"
    )
    display += (
        "-|-".join(["-" * max_lens[keep_index] for keep_index in keep_indexes]) + "\n"
    )
    for format_line in format_lines:
        display += (
            " | ".join(
                [
                    "{:<{}s}".format(format_line[keep_index], max_lens[keep_index])
                    for keep_index in keep_indexes
                ]
            )
            + "\n"
        )
    return display


def build_tree_from_profiler(prof):
    # 以 record_function 名称为 key，构建树
    tree = OrderedDict()
    events = [
        evt
        for evt in prof.key_averages()
        if "." in evt.key or evt.key in ["features", "classifier", "avgpool"]
    ]
    for evt in events:
        path = evt.key.split(".")
        current = tree
        for i, name in enumerate(path):
            if name not in current:
                current[name] = OrderedDict()
            if i == len(path) - 1:
                current[name][None] = Measure(
                    self_cpu_total=evt.self_cpu_time_total,
                    cpu_total=evt.cpu_time_total,
                    self_cuda_total=getattr(evt, "self_cuda_time_total", None),
                    cuda_total=getattr(evt, "cuda_time_total", None),
                    self_cpu_memory=getattr(evt, "self_cpu_memory_usage", None),
                    cpu_memory=getattr(evt, "cpu_memory_usage", None),
                    self_cuda_memory=getattr(evt, "self_cuda_memory_usage", None),
                    cuda_memory=getattr(evt, "cuda_memory_usage", None),
                    occurrences=evt.count,
                )
            current = current[name]
    return tree


def add_hooks(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        def pre_forward_hook(mod, inp, name=full_name):
            mod._rf = record_function(name)
            mod._rf.__enter__()

        def post_forward_hook(mod, inp, out):
            if hasattr(mod, "_rf"):
                mod._rf.__exit__(None, None, None)

        child.register_forward_pre_hook(pre_forward_hook)
        child.register_forward_hook(post_forward_hook)
        add_hooks(child, full_name)


class TorchModuleProfiler:
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.use_cuda = use_cuda
        self.stats = {}

    def _add_hooks(self, module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            def pre_forward_hook(mod, inp, name=full_name):
                mod._rf = record_function(name)
                mod._rf.__enter__()

            def post_forward_hook(mod, inp, out):
                if hasattr(mod, "_rf"):
                    mod._rf.__exit__(None, None, None)

            child.register_forward_pre_hook(pre_forward_hook)
            child.register_forward_hook(post_forward_hook)
            self._add_hooks(child, full_name)

    def profile(self, x, steps=1):
        self._add_hooks(self.model)
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(device)
        x = x.to(device)
        activities = [ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities, record_shapes=True, profile_memory=True
        ) as prof:
            for _ in range(steps):
                self.model(x)
                prof.step()
        self._parse_stats(prof)

    def _parse_stats(self, prof):
        self.stats = {}
        for evt in prof.key_averages():
            if "." in evt.key or evt.key in ["features", "classifier", "avgpool"]:
                self.stats[evt.key] = {
                    "self_cpu": evt.self_cpu_time_total,
                    "cpu_total": evt.cpu_time_total,
                    "self_cuda": getattr(evt, "self_cuda_time_total", 0),
                    "cuda_total": getattr(evt, "cuda_time_total", 0),
                    "self_cpu_mem": getattr(evt, "self_cpu_memory_usage", 0),
                    "cpu_mem": getattr(evt, "cpu_memory_usage", 0),
                    "self_cuda_mem": getattr(evt, "self_cuda_memory_usage", 0),
                    "cuda_mem": getattr(evt, "cuda_memory_usage", 0),
                    "calls": evt.count,
                }

    @staticmethod
    def _fmt_time(t):
        if t >= 1e6:
            return f"{t/1e6:8.3f}ms"
        elif t >= 1e3:
            return f"{t/1e3:8.3f}us"
        elif t > 0:
            return f"{t:8.3f}ns"
        else:
            return "        "

    @staticmethod
    def _fmt_mem(m):
        if m >= 1024 * 1024:
            return f"{m/1024/1024:8.2f} Mb"
        elif m >= 1024:
            return f"{m/1024:8.2f} Kb"
        elif m > 0:
            return f"{m:8.0f} b"
        else:
            return "        "

    def print_tree(self):
        print(
            "Module       | Self CPU total | CPU total | Self CUDA total | CUDA total | Self CPU Mem | CPU Mem | Self CUDA Mem | CUDA Mem  | Number of Calls"
        )
        print(
            "-------------|----------------|-----------|-----------------|------------|--------------|---------|---------------|-----------|----------------"
        )
        self._print_tree(self.model)

    def _print_tree(self, module, prefix="", level=0, is_last=True):
        name = prefix.split(".")[-1] if prefix else module._get_name()
        key = prefix if prefix else module._get_name()
        if level == 0:
            tree_prefix = ""
        else:
            tree_prefix = "│   " * (level - 1)
            tree_prefix += "└── " if is_last else "├── "
        s = self.stats.get(key, {})
        print(
            f"{tree_prefix}{name:<12} |"
            f"{self._fmt_time(s.get('self_cpu',0))} |"
            f"{self._fmt_time(s.get('cpu_total',0))} |"
            f"{self._fmt_time(s.get('self_cuda',0))} |"
            f"{self._fmt_time(s.get('cuda_total',0))} |"
            f"{self._fmt_mem(s.get('self_cpu_mem',0))} |"
            f"{self._fmt_mem(s.get('cpu_mem',0))} |"
            f"{self._fmt_mem(s.get('self_cuda_mem',0))} |"
            f"{self._fmt_mem(s.get('cuda_mem',0))} |"
            f"{s.get('calls',''):>8}"
        )
        children = list(module.named_children())
        for i, (child_name, child) in enumerate(children):
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._print_tree(
                child, child_prefix, level + 1, is_last=(i == len(children) - 1)
            )
