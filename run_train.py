from train_main import run_training_pipeline

if __name__ == "__main__":
    # === 训练配置 ===
    config = {
        # 数据路径 —— 修改成你本地的数据集路径
        "data_root": "",   # 未给
        "save_root": "./experiment_demo",       # 输出目录，自动创建
        "seed": 42,
        "device": None,  # 自动选择 GPU/CPU

        # 模型结构参数
        "point_input_channels": 0,
        "point_out_channels": 128,
        "image_backbone": "resnet18",
        "image_pretrained": False,
        "num_classes": 3,
        "num_proposals": 256,

        # 训练集与验证集划分
        "train_split": "training",
        "val_split": "validation",
        "classes": ["Car", "Cyclist", "Pedestrian"],

        # 训练超参数
        "num_epochs": 5,           
        "batch_size": 2,           
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "iou_pos_thresh": 0.6,
        "iou_neg_thresh": 0.45,
        "score_threshold": 0.3,

        # 恢复训练（可选）
        "resume_checkpoint": None
    }

    # === 开始训练 ===
    history = run_training_pipeline(config)
    print("训练完成，history keys:", list(history.keys()))
