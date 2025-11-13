# tensorrt & cudnn环境
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/system/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# hugging face镜像  
export HF_ENDPOINT=https://hf-mirror.com
# BladeDISC调试选项  
export TORCH_BLADE_DEBUG_LOG=on
