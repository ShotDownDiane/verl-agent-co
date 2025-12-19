#!/bin/bash
# ALFWorld 数据设置脚本
# 使用方法: bash setup_alfworld_data.sh [数据目录]
# 如果不指定目录，默认使用 ~/.cache/alfworld

set -e

# 设置数据目录
ALFWORLD_DATA=${1:-${ALFWORLD_DATA:-~/.cache/alfworld}}
ALFWORLD_DATA=$(eval echo $ALFWORLD_DATA)

echo "=========================================="
echo "ALFWorld 数据设置脚本"
echo "=========================================="
echo "数据目录: $ALFWORLD_DATA"
echo "=========================================="

# 进入数据目录
cd "$ALFWORLD_DATA"

# 创建必要的子目录
mkdir -p detectors
mkdir -p logic

# 检查并解压文件
echo ""
echo "步骤 1: 解压 JSON 数据文件..."
if [ -f "json_2.1.2_json.zip" ]; then
    unzip -q -o json_2.1.2_json.zip
    echo "✓ JSON 数据文件已解压"
elif [ -f "json_2.1.1_json.zip" ]; then
    unzip -q -o json_2.1.1_json.zip
    echo "✓ JSON 数据文件已解压"
else
    echo "✗ 未找到 json_2.1.1_json.zip 或 json_2.1.2_json.zip"
fi

echo ""
echo "步骤 2: 解压 PDDL 文件..."
if [ -f "json_2.1.1_pddl.zip" ]; then
    unzip -q -o json_2.1.1_pddl.zip
    echo "✓ PDDL 文件已解压"
else
    echo "✗ 未找到 json_2.1.1_pddl.zip"
fi

echo ""
echo "步骤 3: 解压 TextWorld PDDL 文件..."
if [ -f "json_2.1.2_tw-pddl.zip" ]; then
    unzip -q -o json_2.1.2_tw-pddl.zip
    echo "✓ TextWorld PDDL 文件已解压"
else
    echo "✗ 未找到 json_2.1.2_tw-pddl.zip"
fi

echo ""
echo "步骤 4: 移动 MaskRCNN 模型..."
if [ -f "mrcnn_alfred_objects_sep13_004.pth" ]; then
    if [ ! -f "detectors/mrcnn_alfred_objects_sep13_004.pth" ]; then
        mv mrcnn_alfred_objects_sep13_004.pth detectors/
        echo "✓ MaskRCNN 模型已移动到 detectors 目录"
    else
        echo "✓ MaskRCNN 模型已在 detectors 目录"
    fi
else
    echo "✗ 未找到 mrcnn_alfred_objects_sep13_004.pth"
fi

# 检查并复制 logic 文件
echo ""
echo "步骤 5: 设置 logic 文件..."
# 从 alfworld 包中复制 logic 文件（如果存在）
if [ -f "/root/miniconda3/lib/python3.12/site-packages/alfworld/data/alfred.pddl" ]; then
    if [ ! -f "logic/alfred.pddl" ]; then
        cp /root/miniconda3/lib/python3.12/site-packages/alfworld/data/alfred.pddl logic/
        echo "✓ alfred.pddl 已复制到 logic 目录"
    else
        echo "✓ alfred.pddl 已存在"
    fi
    if [ ! -f "logic/alfred.twl2" ]; then
        cp /root/miniconda3/lib/python3.12/site-packages/alfworld/data/alfred.twl2 logic/
        echo "✓ alfred.twl2 已复制到 logic 目录"
    else
        echo "✓ alfred.twl2 已存在"
    fi
else
    # 尝试从解压后的目录查找
    if [ -d "json_2.1.1" ] && [ -f "json_2.1.1/alfred.pddl" ]; then
        if [ ! -f "logic/alfred.pddl" ]; then
            cp json_2.1.1/alfred.pddl logic/ 2>/dev/null || true
        fi
    fi
    if [ -d "json_2.1.1" ] && [ -f "json_2.1.1/alfred.twl2" ]; then
        if [ ! -f "logic/alfred.twl2" ]; then
            cp json_2.1.1/alfred.twl2 logic/ 2>/dev/null || true
        fi
    fi
    if [ -f "logic/alfred.pddl" ] && [ -f "logic/alfred.twl2" ]; then
        echo "✓ logic 文件已存在"
    else
        echo "⚠ 警告: 可能需要手动复制 logic 文件"
    fi
fi

# 验证目录结构
echo ""
echo "=========================================="
echo "验证目录结构..."
echo "=========================================="

check_dir() {
    if [ -d "$1" ]; then
        echo "✓ $1 存在"
        return 0
    else
        echo "✗ $1 不存在"
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo "✓ $1 存在"
        return 0
    else
        echo "✗ $1 不存在"
        return 1
    fi
}

check_dir "$ALFWORLD_DATA/json_2.1.1"
check_dir "$ALFWORLD_DATA/json_2.1.1/train"
check_dir "$ALFWORLD_DATA/json_2.1.1/valid_seen"
check_dir "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
check_dir "$ALFWORLD_DATA/logic"
check_file "$ALFWORLD_DATA/logic/alfred.pddl"
check_file "$ALFWORLD_DATA/logic/alfred.twl2"
check_dir "$ALFWORLD_DATA/detectors"
check_file "$ALFWORLD_DATA/detectors/mrcnn_alfred_objects_sep13_004.pth"

echo ""
echo "=========================================="
echo "设置完成！"
echo "=========================================="
echo "数据目录: $ALFWORLD_DATA"
echo ""
echo "现在您可以运行训练脚本："
echo "  cd /root/autodl-tmp/verl-agent-co"
echo "  bash examples/ppo_trainer/run_alfworld.sh"
echo "=========================================="

