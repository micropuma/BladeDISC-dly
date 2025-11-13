import time
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# ======== 1. 加载模型与分词器 ========
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
print("[INFO] Loading model and tokenizer from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda().eval()

# ======== 2. 自定义分词函数 ========
def plain_tokenizer(inputs_str, return_tensors):
    """将文本转为 GPU 上的张量"""
    inputs = tokenizer(inputs_str, return_tensors=return_tensors, padding=True)
    inputs = dict(map(lambda x: (x[0], x[1].cuda()), inputs.items()))
    return (
        inputs["input_ids"].cuda(),
        inputs["attention_mask"].cuda(),
        inputs["token_type_ids"].cuda(),
    )

# ======== 3. 自定义 pipeline ========
class PlainTextClassificationPipeline(TextClassificationPipeline):
    def _forward(self, model_inputs):
        return self.model(*model_inputs)

print("[INFO] Building sentiment analysis pipeline...")
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=plain_tokenizer,
    pipeline_class=PlainTextClassificationPipeline,
    device=0,
)

# ======== 4. 测试推理 ========
input_strs = [
    "We are very happy to show you the story.",
    "We hope you don't hate it.",
]

results = classifier(input_strs)
for inp_str, result in zip(input_strs, results):
    print(inp_str)
    print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")

# ======== 5. 使用 BladeDISC 优化模型 ========
import torch_blade

print("[INFO] Optimizing model with TorchBlade...")
inputs_str = "Hey, the cat is cute."
inputs = plain_tokenizer(inputs_str, return_tensors="pt")

torch_config = torch_blade.config.Config()
torch_config.enable_mlir_amp = False  # 禁用混合精度

with torch.no_grad(), torch_config:
    optimized_ts = torch_blade.optimize(
        model, allow_tracing=True, model_inputs=tuple(inputs)
    )

# 保存优化后的 TorchScript 模型
torch.jit.save(optimized_ts, "opt.disc.pt")
print("[INFO] Optimized TorchScript model saved to opt.disc.pt")

# ======== 6. Benchmark 测试函数 ========
@torch.no_grad()
def benchmark(model, inputs, num_iters=1000):
    """平均推理时间 (毫秒)"""
    # 预热
    for _ in range(10):
        model(*inputs)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        model(*inputs)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / num_iters * 1000.0

def bench_and_report(input_strs):
    print("[INFO] Running benchmark...")
    inputs = plain_tokenizer(input_strs, return_tensors="pt")

    baseline_ms = benchmark(model, inputs)
    disc_ms = benchmark(optimized_ts, inputs)

    print(f"Seqlen: {[len(s) for s in input_strs]}")
    print(f"Baseline: {baseline_ms:.4f} ms")
    print(f"BladeDISC: {disc_ms:.4f} ms")
    print(f"BladeDISC speedup: {baseline_ms / disc_ms:.2f}x")

# ======== 7. 开始性能对比 ========
bench_and_report(input_strs)

print("[INFO] Done.")
