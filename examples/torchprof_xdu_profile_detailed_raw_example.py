# torchprof_xdu_profile_detailed_raw_alexnet_example.py

import torch
import torchvision.models as models

from ..torchprof_xdu_profile_detailed import ProfileDetailed
from ..torchprof_xdu_display_detailed import get_raw_measure_dict_from_profiler_data, \
    display_extracted_measure_dict, simple_value_formatter


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