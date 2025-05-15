# torchprof_xdu_profile_detailed_raw_alexnet_example.py

import torch
import torchvision.models as models

from ..torchprof_xdu_profile_detailed import ProfileDetailed
from ..torchprof_xdu_display_detailed import get_raw_measure_dict_from_profiler_data

def display_extracted_measure_dict(measure_dict, measure_name, is_averaged=False, unit_converter=None, top_n=100, is_sort=False):
    """
    封装一个函数来打印从 get_raw_measure_dict_from_profiler_data 返回的字典。

    Args:
        measure_dict (dict): {module_path_str: value} 字典。
        measure_name (str): 正在显示的指标名称。
        is_averaged (bool): 指标值是否是平均值。
        unit_converter (function, optional): 一个函数，用于将原始值转换为更易读的单位和字符串。
        top_n (int): 最多显示的条目数量。
        is_sort (bool): 是否按值排序。
    """
    value_type_str = "(average per call)" if is_averaged else "(sum over all calls)"
    if measure_name in ["Parameters", "Calls"]: # These are not averaged or summed in the same way
        value_type_str = ""

    print(f"\n'{measure_name}' for each profiled module {value_type_str}:")

    if not measure_dict:
        print(f"  No data found for '{measure_name}'.")
        if isinstance(measure_dict, dict) and not measure_dict:
             print("   (The dictionary returned by get_raw_measure_dict_from_profiler_data was empty).")
        return

    # 为了更有用地显示，可以按值排序 (降序)
    # 过滤掉 None 值以便排序，或者将 None 视为最小值
    valid_items = {k: v for k, v in measure_dict.items() if v is not None}
    if is_sort:
        valid_items = sorted(
            valid_items.items(),
            key=lambda item: item[1],
            reverse=True
        )
    else:
        valid_items = sorted(
            valid_items.items(),
            key=lambda item: item[0],
            reverse=False
        )

    count = 0
    for module_path_str, value in valid_items:
        display_value = str(value) # 默认显示
        if unit_converter:
            display_value = unit_converter(value, measure_name) # 使用转换器格式化

        print(f"  '{module_path_str}': {display_value}")
        count += 1
        if count >= top_n and len(valid_items) > top_n:
            print(f"  ... and {len(valid_items) - count} more modules.")
            break
    
    if not valid_items and measure_dict: # 如果所有值都是 None
        print(f"  All values for '{measure_name}' were None or filtered out.")
    elif not measure_dict: # 如果原始字典就是空的
        print(f"  No modules found with data for '{measure_name}'.")


def simple_value_formatter(value, measure_name):
    """根据指标名称对原始值进行简单格式化和单位转换。"""
    if value is None:
        return "None"
    if "total" in measure_name.lower() and isinstance(value, (int, float)):
        return f"{value / 1_000:.3f} us" # 假设原始时间是纳秒
    elif "Mem" in measure_name and isinstance(value, (int, float)):
        return f"{value / (1024*1024):.3f} MB" # 假设原始内存是字节
    elif "FLOPs" == measure_name and isinstance(value, (int, float)):
        return f"{value / 1e9:.3f} GFLOPs"
    elif "Parameters" == measure_name or "Calls" == measure_name:
        return str(int(value)) if isinstance(value, (int,float)) else str(value)
    return str(value) # 其他情况直接转字符串

def main():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 加载预训练模型
    model = models.alexnet(weights=None)
    model = model.to(device)
    model.eval()

    # 准备输入数据
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    num_runs = 30  # 多跑几次
    print("开始使用 ProfileDetailed 进行性能分析...")
    with ProfileDetailed(
        model,
        enabled=True,
        use_cuda=(device.type == "cuda"),
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)

    print("-" * 30)
    print("默认树状结构输出 (包含FLOPs和参数):")
    print("-" * 30)
    print(prof.display(show_events=False))

    # CPU total
    print("\n--- Extracting specific measures from prof.raw() ---")
    raw_data_output = prof.raw()

    if raw_data_output:
        # --- 演示提取 "CPU total" (总和) ---
        cpu_total_sum_dict = get_raw_measure_dict_from_profiler_data(
            raw_data_output,
            "CPU total",
            average_over_calls=False
        )
        display_extracted_measure_dict(cpu_total_sum_dict, "CPU total (Sum)",
                                       is_averaged=False, unit_converter=simple_value_formatter)

        # --- 演示提取 "CPU total" (平均值) ---
        cpu_total_avg_dict = get_raw_measure_dict_from_profiler_data(
            raw_data_output,
            "CPU total",
            average_over_calls=True
        )
        display_extracted_measure_dict(cpu_total_avg_dict, "CPU total (Avg/call)",
                                       is_averaged=True, unit_converter=simple_value_formatter)

        # --- 演示提取 "Parameters" (average_over_calls 对它无影响) ---
        params_dict = get_raw_measure_dict_from_profiler_data(
            raw_data_output,
            "Parameters",
            average_over_calls=False # 或 True，结果相同
        )
        display_extracted_measure_dict(params_dict, "Parameters",
                                       unit_converter=simple_value_formatter)
        
        # --- 演示提取 "Calls" (average_over_calls 对它无影响) ---
        calls_dict = get_raw_measure_dict_from_profiler_data(
            raw_data_output,
            "Calls",
            average_over_calls=False # 或 True，结果相同
        )
        display_extracted_measure_dict(calls_dict, "Calls",
                                       unit_converter=simple_value_formatter)

        # --- 演示提取 "FLOPs" (平均值) ---
        flops_avg_dict = get_raw_measure_dict_from_profiler_data(
            raw_data_output,
            "FLOPs",
            average_over_calls=True
        )
        display_extracted_measure_dict(flops_avg_dict, "FLOPs (Avg/call)",
                                       is_averaged=True, unit_converter=simple_value_formatter)
        
        if device.type == "cuda":
            # --- 演示提取 "Self CUDA total" (平均值) ---
            cuda_self_avg_dict = get_raw_measure_dict_from_profiler_data(
                raw_data_output,
                "Self CUDA total",
                average_over_calls=True
            )
            display_extracted_measure_dict(cuda_self_avg_dict, "Self CUDA total (Avg/call)",
                                           is_averaged=True, unit_converter=simple_value_formatter)

    else:
        print("Failed to get raw profiling data (prof.raw() returned None).")

if __name__ == "__main__":
    main()