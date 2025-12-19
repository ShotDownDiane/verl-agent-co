# Flash Attention 编译安装指南

## 关键注意事项

### 1. 环境要求

- **CUDA 版本**: >= 12.1 (推荐 12.4+)
- **PyTorch 版本**: 需要与 CUDA 版本匹配（推荐 2.6.0+）
- **Python 版本**: >= 3.9 (推荐 3.10)
- **cuDNN**: >= 9.8.0

### 2. 必须使用 `--no-build-isolation` 参数

这是最重要的！Flash Attention 编译时必须使用此参数：

```bash
pip install flash-attn --no-build-isolation
```

或者如果使用 uv：

```bash
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

### 3. 安装构建依赖

在编译前确保安装必要的构建工具：

```bash
# 安装 ninja（加速编译）
pip install ninja

# 升级构建工具
pip install --upgrade pip wheel setuptools packaging
```

### 4. 设置环境变量

```bash
# 设置 CUDA 路径（根据实际安装路径调整）
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置编译线程数（根据 CPU 核心数调整，不要设置太大避免内存不足）
export MAX_JOBS=32
```

### 5. 版本兼容性

根据项目配置，推荐使用预编译的 wheel 文件（如果可用）：

```bash
# 对于 CUDA 12.4 + PyTorch 2.6.0 + Python 3.10
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 6. 常见问题解决

#### 问题 1: 编译时间过长
- **解决方案**: 使用预编译的 wheel 文件，或减少 `MAX_JOBS` 值

#### 问题 2: 内存不足
- **解决方案**: 减少 `MAX_JOBS` 环境变量值（例如改为 8 或 16）

#### 问题 3: CUDA 头文件找不到
- **解决方案**: 
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH=$CUDA_HOME/bin:$PATH
  ```

#### 问题 4: 缺少 cuda_bf16.h
- **解决方案**: 确保 CUDA 版本 >= 11.0，并正确设置 CUDA_HOME

#### 问题 5: PyTorch 版本不匹配
- **解决方案**: 先安装匹配的 PyTorch，再安装 flash-attn
  ```bash
  pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip3 install flash-attn --no-build-isolation
  ```

### 7. 完整安装流程（推荐）

```bash
# 1. 设置环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=32

# 2. 安装构建依赖
pip install ninja wheel packaging setuptools --upgrade

# 3. 安装匹配的 PyTorch
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 Flash Attention
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

# 5. 验证安装
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 8. 检查安装是否成功

```bash
python -c "from flash_attn import flash_attn_func; print('Flash Attention installed successfully!')"
```

### 9. 项目特定要求

根据 `verl-agent-co` 项目的要求：

- **推荐版本**: Flash Attention 2.7.4.post1
- **必须参数**: `--no-build-isolation`
- **兼容性**: 需要与 PyTorch 2.6.0 和 CUDA 12.4 兼容

### 10. 如果编译失败

1. 检查 CUDA 和 PyTorch 版本是否匹配
2. 确保所有环境变量正确设置
3. 尝试使用预编译的 wheel 文件
4. 减少 `MAX_JOBS` 值重试
5. 查看完整错误日志定位问题

