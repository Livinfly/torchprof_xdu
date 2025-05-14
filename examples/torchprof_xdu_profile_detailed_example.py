import torch
import torchvision.models as models
from ..torchprof_xdu_profile_detailed import ProfileDetailed


def main():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 加载预训练模型 (使用 ResNet18 作为例子，AlexNet 也可以)
    # weights=None 表示随机初始化，如果需要预训练权重，可以使用 weights=models.ResNet18_Weights.DEFAULT
    model = models.alexnet(weights=None)
    model = model.to(device)
    model.eval() # 设置为评估模式，这对于分析很重要，可以关闭 dropout 等

    # 准备输入数据
    # 对于 ResNet18，通常输入是 (N, 3, 224, 224)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    print("=" * 40)
    print("开始使用 ProfileDetailed 进行性能分析...")
    print("=" * 40 + "\n")

    # 使用 ProfileDetailed 进行性能分析
    with ProfileDetailed(
        model,
        enabled=True,
        use_cuda=(device.type == "cuda"),
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for _ in range(30):  # 30
                _ = model(input_tensor)

    # 打印分析结果 - 展示详细版本的各种显示选项

    print("-" * 30)
    print("默认树状结构输出 (包含FLOPs和参数):")
    print("-" * 30)
    print(prof.display(show_events=False))

    # print("-" * 30)
    # print("默认树状结构输出 (包含FLOPs和参数):")
    # print("-" * 30)
    # print(prof.display(show_events=True))

    # print("\n" + "-" * 30)
    # print("按 'FLOPs' 排序 (降序), 显示全部:")
    # print("-" * 30)
    # # 注意：sort_by 的值需要与 torchprof_xdu_display_detailed.py 中的 SORT_BY_MAP 键匹配
    # print(prof.display(sort_by="FLOPs", top_k=-1))

    # print("\n" + "-" * 30)
    # print("按 'Self CPU total' 排序, 显示 Top 5:")
    # print("-" * 30)
    # print(prof.display(sort_by="Self CPU total", top_k=5))

    # if device.type == "cuda":
    #     print("\n" + "-" * 30)
    #     print("按 'Self CUDA total' 排序, 显示 Top 5:")
    #     print("-" * 30)
    #     print(prof.display(sort_by="Self CUDA total", top_k=5))

    #     print("\n" + "-" * 30)
    #     print("显示详细事件, 按 'CUDA total' 排序, 显示 Top 3 事件:")
    #     print("-" * 30)
    #     print(prof.display(show_events=True, sort_by="CUDA total", top_k=3))

    # print("\n" + "-" * 30)
    # print("按 'Parameters' 排序, 显示 Top 7 的模块:")
    # print("-" * 30)
    # print(prof.display(sort_by="Parameters", top_k=7))

    # print("\n" + "-" * 30)
    # print("按 'CPU Mem' 排序 (可能是增量或分配), 显示 Top 5:")
    # print("-" * 30)
    # print(prof.display(sort_by="CPU Mem", top_k=5))

    print("\n" + "=" * 40)
    print("性能分析结束。")
    print("=" * 40)

if __name__ == "__main__":
    main()