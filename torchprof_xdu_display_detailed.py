import torch
from collections import OrderedDict, namedtuple

# 扩展 Measure 以包含参数数量和 FLOPs
Measure = namedtuple(
    "Measure",
    [
        "self_cpu_total", "cpu_total",
        "self_cuda_total", "cuda_total",
        "self_cpu_memory", "cpu_memory",
        "self_cuda_memory", "cuda_memory",
        "occurrences",
        "params", "flops"
    ],
)

# 用于排序的指标名称到 Measure 字段名称的映射
SORT_BY_MAP = {
    "Self CPU total": "self_cpu_total", "CPU total": "cpu_total",
    "Self CUDA total": "self_cuda_total", "CUDA total": "cuda_total",
    "Self CPU Mem": "self_cpu_memory", "CPU Mem": "cpu_memory",
    "Self CUDA Mem": "self_cuda_memory", "CUDA Mem": "cuda_memory",
    "FLOPs": "flops", "Parameters": "params", "Calls": "occurrences",
    # Raw field names also accepted for convenience
    "self_cpu_total": "self_cpu_total", "cpu_total": "cpu_total",
    "self_cuda_total": "self_cuda_total", "cuda_total": "cuda_total",
    "self_cpu_memory": "self_cpu_memory", "cpu_memory": "cpu_memory",
    "self_cuda_memory": "self_cuda_memory", "cuda_memory": "cuda_memory",
    "flops": "flops", "params": "params", "occurrences": "occurrences",
}


def _format_time(t):
    if t is None: return ""
    if t == 0: return "0.000ns"
    unit = "ns"
    if abs(t) >= 1000: t /= 1000; unit = "us"
    if abs(t) >= 1000: t /= 1000; unit = "ms"
    if abs(t) >= 1000: t /= 1000; unit = "s"
    return f"{t:.3f}{unit}"

def _format_memory(m):
    if m is None: return ""
    if m == 0: return "0 b"
    sign = "-" if m < 0 else ""
    m = abs(m)
    unit = "b"
    if m >= 1024: m /= 1024; unit = "Kb"
    if m >= 1024: m /= 1024; unit = "Mb" # Already divided by 1024 for Kb
    if m >= 1024: m /= 1024; unit = "Gb" # Already divided by 1024*1024 for Mb
    return f"{sign}{m:.2f} {unit}"

def _format_flops(f):
    if f is None or f == 0: return "0 FLOPs"
    unit = ""
    if abs(f) >= 1e3: f /= 1e3; unit = "K"
    if abs(f) >= 1e3: f /= 1e3; unit = "M"
    if abs(f) >= 1e3: f /= 1e3; unit = "G"
    if abs(f) >= 1e3: f /= 1e3; unit = "T"
    return f"{f:.2f}{unit}FLOPs"

def _format_params(p):
    if p is None or p == 0: return "0"
    unit = ""
    if abs(p) >= 1e3: p /= 1e3; unit = "K"
    if abs(p) >= 1e3: p /= 1e3; unit = "M"
    if abs(p) >= 1e3: p /= 1e3; unit = "B"
    return f"{p:.2f}{unit}"

def _format_measure_tuple(measure):
    if not measure:
        return Measure(*([""] * len(Measure._fields)))
    return Measure(
        self_cpu_total=_format_time(measure.self_cpu_total),
        cpu_total=_format_time(measure.cpu_total),
        self_cuda_total=_format_time(measure.self_cuda_total),
        cuda_total=_format_time(measure.cuda_total),
        self_cpu_memory=_format_memory(measure.self_cpu_memory),
        cpu_memory=_format_memory(measure.cpu_memory),
        self_cuda_memory=_format_memory(measure.self_cuda_memory),
        cuda_memory=_format_memory(measure.cuda_memory),
        occurrences=str(measure.occurrences) if measure.occurrences is not None else "",
        params=_format_params(measure.params),
        flops=_format_flops(measure.flops)
    )

def _flatten_tree(t, depth=0):
    flat = []
    for name, st_content in t.items():
        # st_content is either an OrderedDict (for children) or a Measure object (for the node itself)
        # The structure is: tree[module_name] = OrderedDict()
        # then, tree[module_name][None] = MeasureObj OR tree[module_name][event_name] = {None: MeasureObj}
        measures_obj = st_content.pop(None, None) if isinstance(st_content, OrderedDict) else None
        flat.append([depth, name, measures_obj]) # name is module name or event name
        if isinstance(st_content, OrderedDict) and st_content: # If there are children (nested OrderedDict)
             flat.extend(_flatten_tree(st_content, depth=depth + 1))
    return flat

def _build_measure_tuple(events_for_module, occurrences, module_params, module_flops_from_event_placeholder=0):
    self_cpu_total_sum, cpu_total_sum, self_cuda_total_sum, cuda_total_sum = 0,0,0,0
    self_cpu_memory_sum, cpu_memory_sum, self_cuda_memory_sum, cuda_memory_sum = 0,0,0,0
    total_flops_from_events = 0

    if not events_for_module or not any(events_for_module): # occurrences can be 0
        return Measure(0,0,0,0,0,0,0,0, occurrences, module_params, 0)

    all_single_events = [evt for event_list in events_for_module for evt in event_list]
    if not all_single_events and occurrences == 0 : # Only return all zeros if no calls and no events
         return Measure(0,0,0,0,0,0,0,0, 0, module_params, 0)
    if not all_single_events and occurrences > 0: # Module called but produced no profiler events (e.g. empty module)
         return Measure(0,0,0,0,0,0,0,0, occurrences, module_params, 0)


    for event in all_single_events:
        self_cpu_total_sum += getattr(event, "self_cpu_time_total", 0)
        cpu_total_sum += getattr(event, "cpu_time_total", 0)
        self_cuda_total_sum += getattr(event, "self_cuda_time_total", 0)
        cuda_total_sum += getattr(event, "cuda_time_total", 0)
        self_cpu_memory_sum += getattr(event, "self_cpu_memory_usage", 0)
        cpu_memory_sum += getattr(event, "cpu_memory_usage", 0)
        self_cuda_memory_sum += getattr(event, "self_cuda_memory_usage", 0)
        cuda_memory_sum += getattr(event, "cuda_memory_usage", 0)
        total_flops_from_events += getattr(event, "flops", 0)

    return Measure(
        self_cpu_total=self_cpu_total_sum, cpu_total=cpu_total_sum,
        self_cuda_total=self_cuda_total_sum if self_cuda_total_sum > 0 or any(hasattr(e, "self_cuda_time_total") for e in all_single_events) else None,
        cuda_total=cuda_total_sum if cuda_total_sum > 0 or any(hasattr(e, "cuda_time_total") for e in all_single_events) else None,
        self_cpu_memory=self_cpu_memory_sum if self_cpu_memory_sum != 0 or any(hasattr(e, "self_cpu_memory_usage") for e in all_single_events) else None,
        cpu_memory=cpu_memory_sum if cpu_memory_sum != 0 or any(hasattr(e, "cpu_memory_usage") for e in all_single_events) else None,
        self_cuda_memory=self_cuda_memory_sum if self_cuda_memory_sum != 0 or any(hasattr(e, "self_cuda_memory_usage") for e in all_single_events) else None,
        cuda_memory=cuda_memory_sum if cuda_memory_sum != 0 or any(hasattr(e, "cuda_memory_usage") for e in all_single_events) else None,
        occurrences=occurrences, params=module_params, flops=total_flops_from_events
    )

def group_by_key(event_lists, keyfn): # Renamed from group_by to avoid conflict if imported elsewhere
    event_groups = OrderedDict()
    for event_list_for_one_call in event_lists: # event_lists is list of EventList
        for evt_item in event_list_for_one_call: # evt_item is an actual profiler event object
            key = keyfn(evt_item)
            key_events_group = event_groups.get(key, [])
            key_events_group.append(evt_item)
            event_groups[key] = key_events_group
    return event_groups.items() # returns list of (key, list_of_events_for_that_key)

def traces_to_display_detailed(
    traces, trace_profile_events, trace_module_details,
    show_events=False, paths=None, use_cuda=False, profile_memory=False,
    sort_by=None, top_k=-1, # New arguments for sorting and top-k
    dt=("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " ")
):
    tree = OrderedDict()
    for trace_item in traces:
        path_tuple, leaf, module_obj, module_node_name = trace_item
        current_tree_ptr = tree
        for depth_idx, name_in_path in enumerate(path_tuple, 1):
            if name_in_path not in current_tree_ptr:
                current_tree_ptr[name_in_path] = OrderedDict()
            
            if depth_idx == len(path_tuple): # This node represents the module itself
                should_record_stats = (paths is None and leaf) or \
                                      (paths is not None and path_tuple in paths)
                if should_record_stats:
                    module_event_lists = trace_profile_events.get(path_tuple, [])
                    num_calls = len(module_event_lists)
                    module_params = trace_module_details.get(path_tuple, {}).get("params", 0)

                    if show_events:
                        # Group events by event.key (e.g., 'aten::conv2d')
                        # module_event_lists is list of EventList.
                        for event_key, event_group_actual_events in group_by_key(module_event_lists, lambda e: e.key):
                            # event_group_actual_events is a list of actual event objects for this key
                            event_specific_flops = sum(getattr(e, "flops", 0) for e in event_group_actual_events)
                            # Create a Measure for this specific event group within the module
                            # The 'occurrences' for an event group is how many times that specific op was called
                            measure_for_event_group = _build_measure_tuple(
                                [event_group_actual_events], # Pass as list of list of events
                                len(event_group_actual_events),
                                0, # Params are module-level, not for individual events here
                                event_specific_flops # Pass specific FLOPs for this event group
                            )
                            # Store under module_name -> event_key -> None: Measure
                            if event_key not in current_tree_ptr[name_in_path]:
                                current_tree_ptr[name_in_path][event_key] = OrderedDict()
                            current_tree_ptr[name_in_path][event_key][None] = measure_for_event_group
                    else:
                         # Aggregate all events for this module
                        current_tree_ptr[name_in_path][None] = _build_measure_tuple(
                            module_event_lists, num_calls, module_params
                        )
            current_tree_ptr = current_tree_ptr[name_in_path]

    # Flatten the hierarchical tree into a list of lines for display
    # Each line: [depth, name, measures_obj (raw Measure or None)]
    tree_lines_flat = _flatten_tree(tree)

    # Filter for sortable lines (those with actual Measure objects)
    # and structural lines (hierarchy placeholders, measures_obj is None)
    sortable_lines = [line for line in tree_lines_flat if line[2] is not None]
    
    # Apply sorting and top-k filtering if requested
    if sort_by and sortable_lines:
        sort_by_field = SORT_BY_MAP.get(sort_by)
        if not sort_by_field:
            valid_keys = [k for k, v in SORT_BY_MAP.items() if v in Measure._fields]
            raise ValueError(f"Invalid sort_by key: '{sort_by}'. Valid keys are: {valid_keys}")

        def get_sort_key(item_line):
            measure = item_line[2] # measures_obj
            val = getattr(measure, sort_by_field, 0) # Default to 0 if field missing
            return val if val is not None else 0 # Handle None for the metric itself

        sortable_lines.sort(key=get_sort_key, reverse=True) # Sort descending

        if top_k != -1 and top_k > 0:
            sortable_lines = sortable_lines[:top_k]
        
        processed_tree_lines = sortable_lines # Display only sorted & filtered items
    else:
        processed_tree_lines = tree_lines_flat # Display original tree structure

    if not processed_tree_lines:
        return "No data to display."

    # Prepare formatted lines for the table
    format_lines = []
    has_self_cuda_total, has_cuda_total, has_self_cpu_memory, has_cpu_memory = False, False, False, False
    has_self_cuda_memory, has_cuda_memory, has_flops, has_params = False, False, False, False

    for idx, (depth, name, measures_obj) in enumerate(processed_tree_lines):
        # Calculate tree prefix for indentation
        pre = ""
        if depth > 0:
            # Check continuity with the *next* item in the *current* list (processed_tree_lines)
            # to determine connector type (├─ vs └─)
            is_last_at_this_depth_level = True
            for next_line_depth, _, _ in processed_tree_lines[idx + 1:]:
                if next_line_depth == depth: # Another sibling at same depth follows
                    is_last_at_this_depth_level = False
                    break
                if next_line_depth < depth: # Parent level, so this was the last
                    break
            
            # Vertical bars for parent levels
            ancestor_depths = set(item_depth for item_depth, _, _ in processed_tree_lines[idx + 1:] if item_depth < depth)
            
            temp_pre = dt[2] if is_last_at_this_depth_level else dt[1] # └─ or ├─
            
            for d_idx in range(depth -1, -1, -1): # Iterate from parent depth down to root
                if d_idx in ancestor_depths : # Check if any future line connects through this ancestor depth
                     # More robust check: scan future lines to see if a branch from this d_idx continues
                    parent_continues = any(  pl[0] > d_idx for pl_idx, pl in enumerate(processed_tree_lines[idx+1:]) if all(processed_tree_lines[idx+1+k][0] >=d_idx for k in range(pl_idx+1)) )
                    # Simplified: if any future line has this depth d_idx as part of its path visually
                    # This part is complex to get perfect with arbitrary sorting.
                    # A simpler approach for prefixes if sorted: just use depth for indentation.
                    if any(pl[0] > d_idx for pl in processed_tree_lines[idx+1:] if pl[0] >= d_idx):
                         temp_pre = dt[0] + "  " + temp_pre # │
                    else:
                         temp_pre = dt[3] + "  " + temp_pre # space
                else: # No continuing line from this ancestor depth among remaining items
                    temp_pre = dt[3] + "  " + temp_pre # space

            pre = temp_pre
        else: # depth == 0
            pre = ""


        formatted_measure_values = _format_measure_tuple(measures_obj)
        format_lines.append([pre + name, *formatted_measure_values])

        if measures_obj: # Update flags for dynamic column display
            has_self_cuda_total = has_self_cuda_total or (measures_obj.self_cuda_total is not None and measures_obj.self_cuda_total != 0)
            has_cuda_total = has_cuda_total or (measures_obj.cuda_total is not None and measures_obj.cuda_total != 0)
            has_self_cpu_memory = has_self_cpu_memory or (measures_obj.self_cpu_memory is not None and measures_obj.self_cpu_memory !=0)
            has_cpu_memory = has_cpu_memory or (measures_obj.cpu_memory is not None and measures_obj.cpu_memory !=0)
            has_self_cuda_memory = has_self_cuda_memory or (measures_obj.self_cuda_memory is not None and measures_obj.self_cuda_memory !=0)
            has_cuda_memory = has_cuda_memory or (measures_obj.cuda_memory is not None and measures_obj.cuda_memory !=0)
            has_flops = has_flops or (measures_obj.flops is not None and measures_obj.flops !=0)
            has_params = has_params or (measures_obj.params is not None and measures_obj.params !=0)

    # Define table headings
    # Order of Measure: s_cpu, cpu, s_cuda, cuda, s_cpu_m, cpu_m, s_cuda_m, cuda_m, occ, params, flops
    # Corresponding indices in formatted_measure_values (after name): 0..10
    heading_all = [
        "Module",                     # 0
        "Self CPU total", "CPU total", # 1, 2
        "Self CUDA total", "CUDA total",# 3, 4
        "Self CPU Mem", "CPU Mem",    # 5, 6
        "Self CUDA Mem", "CUDA Mem",   # 7, 8
        "FLOPs",                      # 9 (maps to measure.flops, which is index 10 of formatted_measure_values)
        "Parameters",                 # 10 (maps to measure.params, which is index 9 of formatted_measure_values)
        "Calls"                       # 11 (maps to measure.occurrences, index 8 of formatted_measure_values)
    ]
    
    # Determine which columns to display based on data and settings
    selected_heading_indices = [0, 1, 2] # Module, Self CPU, CPU total always shown if data exists
    if use_cuda:
        if has_self_cuda_total: selected_heading_indices.append(3)
        if has_cuda_total: selected_heading_indices.append(4)
    if profile_memory:
        if has_self_cpu_memory: selected_heading_indices.append(5)
        if has_cpu_memory: selected_heading_indices.append(6)
        if use_cuda:
            if has_self_cuda_memory: selected_heading_indices.append(7)
            if has_cuda_memory: selected_heading_indices.append(8)
    if has_flops: selected_heading_indices.append(9) # FLOPs
    if has_params: selected_heading_indices.append(10) # Parameters
    selected_heading_indices.append(11) # Calls
    selected_heading_indices = sorted(list(set(selected_heading_indices)))

    display_heading = [heading_all[i] for i in selected_heading_indices]
    
    # Map selected_heading_indices to indices in format_lines[j][1:]
    # format_lines[j] = [name, scpu, cpu, scu, cu, scpum, cpum, scum, cum, occ_str, par_str, flo_str]
    # Indices in formatted_measure_values:0,    1,   2,    3,   4,     5,    6,     7,     8,       9,       10
    # Corresponding heading_all index:    1,    2,   3,    4,   5,     6,    7,     8,     11,      10,      9
    
    map_heading_idx_to_data_idx = {
        1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, # Times and Memory
        11:8, # Calls (occurrences)
        10:9, # Parameters
        9:10  # FLOPs
    }

    display_lines_final = []
    for f_line in format_lines:
        current_display_row = [f_line[0]] # Module name with prefix
        for heading_idx in selected_heading_indices:
            if heading_idx == 0: continue # Already took name
            data_idx = map_heading_idx_to_data_idx.get(heading_idx)
            if data_idx is not None:
                 current_display_row.append(f_line[1+data_idx]) # +1 because f_line[0] is name
            # else: current_display_row.append("") # Should not happen if map is correct
        display_lines_final.append(current_display_row)
        
    if not display_lines_final and not display_heading: return "No data to display."
    
    # Calculate max lengths for alignment
    if not display_lines_final and display_heading :
        max_lens = [len(h) for h in display_heading]
    elif not display_lines_final: # Should be caught by "No data to display"
        return "No data to display (empty lines)."
    else:
        # Ensure all rows in display_lines_final have same number of columns as display_heading
        num_cols = len(display_heading)
        for i, row in enumerate(display_lines_final):
            if len(row) != num_cols:
                # This indicates a bug in column selection or data mapping
                # print(f"Warning: Row {i} has {len(row)} cols, expected {num_cols}. Row: {row}")
                # Pad a_line if necessary for zipping
                display_lines_final[i] = (row + [""] * num_cols)[:num_cols]


        max_lens = [max(map(len, col)) for col in zip(*([display_heading] + display_lines_final))]

    header_str = " | ".join(["{:<{}s}".format(display_heading[i], max_lens[i]) for i in range(len(display_heading))])
    separator_str = "-|-".join(["-" * max_lens[i] for i in range(len(display_heading))])
    lines_str = "\n".join(
        [" | ".join(["{:<{}s}".format(line[i], max_lens[i]) for i in range(len(line))]) for line in display_lines_final]
    )
    return f"{header_str}\n{separator_str}\n{lines_str}\n"

def get_raw_measure_dict_from_profiler_data(
        profiler_raw_data,
        target_measure_name: str,
        average_over_calls: bool = False
    ):
    """
    根据 ProfileDetailed.raw() 的输出和指定的指标名称，提取每个被剖析模块的特定指标值。
    此函数依赖于在同一文件中定义的 _build_measure_tuple 函数来聚合每个模块的指标。

    参数:
        profiler_raw_data: ProfileDetailed.raw() 的输出, 期望是一个元组:
                           (traces, trace_profile_events, trace_module_details)。
        target_measure_name (str): 要提取的指标的名称,
                                   应与 SORT_BY_MAP 中的键匹配。
        average_over_calls (bool): 如果为 True, 对于时间、内存、FLOPs等动态指标,
                                   返回每次调用的平均值。否则返回所有调用的总和。
                                   参数数量 (Parameters) 和调用次数 (Occurrences/Calls)
                                   不受此标志影响，总是返回其本身的值。

    返回:
        dict: 一个字典，将模块路径字符串映射到对应的原始指标数值。
              如果发生错误或未找到数据，则返回空字典。
    """
    if profiler_raw_data is None:
        print("Error: profiler_raw_data is None. Cannot extract measures.") # 英文提示
        return {}

    try:
        # Unpack based on ProfileDetailed.raw() structure
        traces, trace_profile_events, trace_module_details = profiler_raw_data
    except (ValueError, TypeError):
        print("Error: profiler_raw_data format is incorrect. " # 英文提示
              "Expected (traces, trace_profile_events, trace_module_details).")
        return {}

    output_dict = {}
    # Map the user-facing measure name to the internal field name of the Measure namedtuple
    measure_field_name = SORT_BY_MAP.get(target_measure_name)

    if not measure_field_name:
        print(f"Warning: Unrecognized measure name '{target_measure_name}'. " # 英文提示
              f"Valid names include: {list(SORT_BY_MAP.keys())}")
        return {}

    # Identify all unique module paths that have been profiled or have details
    all_profiled_module_paths = set(trace_profile_events.keys()) | set(trace_module_details.keys())

    for path_tuple in all_profiled_module_paths:
        module_path_str = ".".join(path_tuple) # Convert path tuple to string for dict key

        module_event_lists = trace_profile_events.get(path_tuple, [])
        num_calls_for_module = len(module_event_lists)
        
        module_static_details = trace_module_details.get(path_tuple, {})
        parameter_count = module_static_details.get("params", 0)
        
        # Use _build_measure_tuple to get an aggregated Measure object for this module
        # This Measure object contains summed values over all calls.
        # _build_measure_tuple 应该在当前文件中定义
        aggregated_module_measures = _build_measure_tuple(
            module_event_lists,
            num_calls_for_module,
            parameter_count
        )
        
        if hasattr(aggregated_module_measures, measure_field_name):
            raw_value = getattr(aggregated_module_measures, measure_field_name)

            # If average is requested and the metric is not 'params' or 'occurrences'
            if average_over_calls and num_calls_for_module > 0 and \
               measure_field_name not in ["params", "occurrences"]:
                if raw_value is not None: # Ensure value is not None before division
                    raw_value /= num_calls_for_module
            
            # 'occurrences' from _build_measure_tuple is already the total number of calls.
            # 'params' is a fixed value per module.
            # So, no special handling needed for them here regarding averaging.

            output_dict[module_path_str] = raw_value
        else:
            # 保持调试信息为英文，或者按需修改
            print(f"Debug: Measure field '{measure_field_name}' (from user input '{target_measure_name}') "
                  f"not found in aggregated measures for module '{module_path_str}'. "
                  f"Aggregated measures: {aggregated_module_measures}")

    return output_dict