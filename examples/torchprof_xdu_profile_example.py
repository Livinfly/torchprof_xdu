import torch
import torchvision.models as models
from ..torchprof_xdu_profile import Profile


def main():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")

    # 加载预训练模型
    model = models.alexnet(weights=None)
    model = model.to(device)

    # 准备输入数据
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # 使用Profile进行性能分析
    with Profile(
        model, enabled=True, use_cuda=(device.type == "cuda"), profile_memory=True
    ) as prof:
        # 运行模型
        with torch.no_grad():
            _ = model(input_tensor)

    # 打印分析结果
    print(prof.display(show_events=False))





if __name__ == "__main__":
    main()