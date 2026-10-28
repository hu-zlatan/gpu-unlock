#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""gpu-unlock 解锁前后对比跑分。

gpu-unlock 声称解除 GeForce 上「FP32 累加的 FP16/BF16/TF32 张量核心只跑半速」的限制。
这里测的就是这几项矩阵乘，加一项注意力（SDPA）实际负载，以及几项不该变化的对照：
FP16 累加（GeForce 本来就是全速）、FP32 CUDA 核心、INT8。

用法:
  python3 bench/gpu_unlock_bench.py --out before.json      # 不带 gpu_unlock.targets= 启动时跑
  python3 bench/gpu_unlock_bench.py --out after.json       # 带 gpu_unlock.targets=all 启动时跑
  python3 bench/gpu_unlock_bench.py --compare before.json after.json

两次要用同一套驱动、PyTorch、功耗墙，跑之前关掉其它占用 GPU 的程序。
单跑一次也能判断：FP16/BF16 GEMM 的实测值超过「锁定状态的理论上限」
（SM 数 × 512 FLOP/周期 × 实测频率）就说明确实解锁了，锁定的卡不可能超过它。
"""
import argparse
import contextlib
import ctypes
import datetime
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# 锁定状态下每个 SM 每周期的稠密张量核心 FLOP（FP32 累加），图灵 / 安培 / Ada 的 GeForce 都一样，解锁后翻倍。
# 例：RTX 3090 是 82 SM × 512 × 1.695 GHz = 71 TFLOPS，与 NVIDIA 白皮书一致。
LOCKED_FLOP_PER_SM_CLK = {"fp16": 512, "bf16": 512, "tf32": 256}

# (键, 显示名, 类别)。capped = gpu-unlock 解除限制的项；ctrl = 不应变化的对照。
TESTS = (
    ("bf16", "BF16 GEMM", "capped"),
    ("fp16", "FP16 GEMM", "capped"),
    ("tf32", "TF32 GEMM", "capped"),
    ("sdpa", "FP16 SDPA 注意力", "capped"),
    ("fp16acc16", "FP16 GEMM FP16累加", "ctrl"),
    ("fp32", "FP32 GEMM CUDA核心", "ctrl"),
    ("int8", "INT8 GEMM", "ctrl"),
)

# PyTorch 2.7 起才能让 cuBLAS 的 FP16 矩阵乘用 FP16 累加。
HAS_FP16_ACC = hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")

# kernel/gpu_unlock/main.c 对每张卡最后打印的结果行。
DECISIVE = re.compile(r"gpu_unlock: (unlocked|still unlocked|unlock.* failed|firmware load failed|no supported card)")


class Skip(Exception):
    pass


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w):
    return s + " " * max(0, w - width(s))


def smi(fields, gpu=None):
    """nvidia-smi 查询一张卡，返回 {字段: 值}；查不到或分不清是哪张卡时返回 None。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid," + ",".join(fields), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [[x.strip() for x in line.split(",")] for line in out.splitlines()]
    rows = [r for r in rows if len(r) == len(fields) + 1]
    matched = [r for r in rows if gpu and r[0].lower().endswith(gpu.lower())]
    rows = matched or rows
    return dict(zip(fields, rows[0][1:])) if len(rows) == 1 else None


class Sampler(threading.Thread):
    """后台每 0.3 s 用 nvidia-smi 采样功耗、SM 频率、温度，记在当前测试名下。"""
    FIELDS = ("power.draw", "clocks.sm", "temperature.gpu")

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.label = None
        self.samples = defaultdict(list)
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(0.3):
            label = self.label
            row = smi(self.FIELDS, self.gpu) if label else None
            if row and label == self.label:
                self.samples[label].append([num(row[f]) for f in self.FIELDS])

    def mean(self, label):
        """[平均功耗 W, 平均 SM 频率 MHz, 平均温度 °C]，没采到的项为 None。"""
        result = []
        for i in range(len(self.FIELDS)):
            values = [s[i] for s in self.samples.get(label, []) if s[i] is not None]
            result.append(statistics.fmean(values) if values else None)
        return result


def kernel_log():
    """本次启动里含 gpu_unlock 的内核日志行；读不到（通常是权限不够）返回 None。"""
    for cmd in (["journalctl", "-k", "-b", "-o", "cat", "--no-pager"], ["dmesg"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        lines = r.stdout.splitlines()
        if r.returncode == 0 and len(lines) > 20:  # 读到的是真日志，而不是一行权限提示
            return [line.strip() for line in lines if "gpu_unlock" in line]
    return None


def unlock_state(lines):
    if lines is None:
        return "读不到内核日志（权限不够），手动执行 sudo journalctl -k -b | grep gpu_unlock 查看"
    decisive = [line for line in lines if DECISIVE.search(line)]
    if not decisive:
        return ("日志里有 gpu_unlock 但没有结果行，看 journalctl -k -b | grep gpu_unlock 的完整输出" if lines
                else "本次启动 gpu_unlock 没有运行（日志里没有它）")
    last = decisive[-1]
    return ("解锁失败：" if "failed" in last or "no supported" in last else "已解锁：") + last


def system_info(dev):
    info = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(dev)}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, sm=p.multi_processor_count, cc=f"{p.major}.{p.minor}",
                    uuid=str(getattr(p, "uuid", "")) or None)
        row = smi(("pci.bus_id", "pci.device_id", "driver_version", "vbios_version", "power.limit"), info["uuid"])
        if row:
            info.update(pci_bus=row["pci.bus_id"], driver=row["driver_version"], vbios=row["vbios_version"],
                        power_limit=num(row["power.limit"]))
            with contextlib.suppress(ValueError):
                info["pci_id"] = f"10de:{int(row['pci.device_id'], 16) >> 16:04x}"
    try:
        cmdline = Path("/proc/cmdline").read_text()
    except OSError:
        cmdline = ""
    m = re.search(r"\bgpu_unlock\.targets=(\S*)", cmdline)
    info["cmdline_targets"] = m.group(1) if m else None
    info["module_loaded"] = Path("/sys/module/gpu_unlock").exists()
    info["kernel_log"] = kernel_log()
    return info


@contextlib.contextmanager
def matmul_mode(tf32=False, fp16_acc=False):
    """临时切换 FP32 矩阵乘是否用 TF32、FP16 矩阵乘是否用 FP16 累加。"""
    fp16_acc = fp16_acc and HAS_FP16_ACC
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if fp16_acc:
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        if fp16_acc:
            torch.backends.cuda.matmul.allow_fp16_accumulation = False


def make_case(key, n, dev):
    """返回 (单次调用的函数, 单次 FLOP 数, 取输出的函数)，不适用时抛 Skip。"""
    torch.manual_seed(0)  # 固定输入，前后两次的输出才能逐位比对
    if key in ("bf16", "tf32") and dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] < 8:
        raise Skip("图灵（RTX 20）没有 BF16/TF32 张量核心")
    if key == "fp16acc16" and not HAS_FP16_ACC:
        raise Skip("需要 PyTorch 2.7 以上")
    if key == "sdpa":
        s = n // 2
        q, k, v = (torch.randn(8, 16, s, 128, device=dev, dtype=torch.float16) for _ in range(3))
        out = {}

        def fn():
            out["o"] = F.scaled_dot_product_attention(q, k, v)
        return fn, 4 * 8 * 16 * s * s * 128, lambda: out["o"]
    if key == "int8":
        a = torch.randint(-128, 128, (n, n), device=dev, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, n), device=dev, dtype=torch.int8)
        try:
            c = torch._int_mm(a, b)
        except (AttributeError, RuntimeError) as error:
            raise Skip(f"这版 PyTorch 不支持 torch._int_mm：{error}") from None
        return lambda: torch._int_mm(a, b, out=c), 2 * n ** 3, lambda: c
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp16acc16": torch.float16}.get(key, torch.float32)
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = torch.empty(n, n, device=dev, dtype=dtype)
    return lambda: torch.mm(a, b, out=c), 2 * n ** 3, lambda: c


def measure(fn, flops, seconds, sync, sampler, label):
    """预热 1 s 后连续测 seconds 秒，返回每段约 0.2 s 的吞吐（T(FL)OPS）列表。"""
    fn()  # 首次调用：选算法、分配 workspace
    sync()
    t = time.perf_counter()
    fn()
    sync()
    reps = max(1, int(0.2 / max(time.perf_counter() - t, 1e-6)))
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end:  # 预热，让频率和温度稳定
        for _ in range(reps):
            fn()
        sync()
    rates = []
    sampler.label = label
    end = time.perf_counter() + seconds
    while not rates or time.perf_counter() < end:
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        rates.append(flops * reps / (time.perf_counter() - t) / 1e12)
    sampler.label = None
    return rates


def digest(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())).hexdigest()[:16]


def locked_ceiling(key, info, clock):
    """锁定状态在实测频率下的理论上限（TFLOPS），算不出时返回 None。"""
    per = LOCKED_FLOP_PER_SM_CLK.get(key)
    if not per or not clock or not info.get("sm"):
        return None
    return info["sm"] * per * clock * 1e6 / 1e12


def run(args):
    dev = torch.device(args.device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("PyTorch 没检测到 CUDA GPU（装的是 CPU 版 PyTorch，或驱动太旧带不动这版 PyTorch？）")
        if dev.index is None:  # set_device 不接受不带编号的 "cuda"
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None  # noqa: E731
    info = system_info(dev)
    print_header(info)

    sampler = Sampler(info.get("uuid"))
    sampler.start()
    results = {}
    print(f"\n矩阵 {args.size}³，每项预热 1 s、测 {args.seconds:g} s：")
    for key, name, kind in TESTS:
        unit = "TOPS" if key == "int8" else "TFLOPS"
        entry = {"name": name, "kind": kind, "unit": unit}
        try:
            with matmul_mode(tf32=key == "tf32", fp16_acc=key == "fp16acc16"):
                fn, flops, output = make_case(key, args.size, dev)
                rates = measure(fn, flops, args.seconds, sync, sampler, key)
                entry.update(value=statistics.median(rates), min=min(rates), max=max(rates),
                             hash=digest(output()))
            entry["power"], entry["clock"], entry["temp"] = sampler.mean(key)
        except Skip as reason:
            entry["skipped"] = str(reason)
        except torch.cuda.OutOfMemoryError:
            entry["skipped"] = "显存不够，试试 --size 4096"
        finally:
            fn = output = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        results[key] = entry
        print(f"  {pad(name, 20)} {format_value(entry)}", flush=True)
    sampler.done.set()

    data = {"info": info, "size": args.size, "seconds": args.seconds, "results": results}
    print_table(data)
    print_verdict(data)
    out = Path(args.out or f"gpu_unlock_bench_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}；换另一种状态（带 / 不带 gpu_unlock.targets=）启动后再跑一次，然后:")
    print(f"  python3 {sys.argv[0]} --compare <解锁前.json> <解锁后.json>")


def format_value(entry):
    if "skipped" in entry:
        return f"跳过：{entry['skipped']}"
    return f"{entry['value']:7.1f} {entry['unit']}"


def print_header(info):
    if "gpu" in info:
        limit = info.get("power_limit")
        print(f"GPU: {info['gpu']}  {info.get('pci_id', '?')}  {info['sm']} SM  算力 {info['cc']}  "
              f"驱动 {info.get('driver', '?')}  功耗墙 {f'{limit:.0f}' if limit else '?'} W")
    print(f"PyTorch {info['torch']}  CUDA {info['cuda']}")
    targets = info["cmdline_targets"]
    print(f"本次启动参数 gpu_unlock.targets: {targets if targets else '没有（不会解锁）'}"
          f"；gpu_unlock 模块{'已' if info['module_loaded'] else '未'}加载")
    print(f"gpu_unlock 状态: {unlock_state(info['kernel_log'])}")


def print_table(data):
    info, results = data["info"], data["results"]
    print(f"\n{pad('测试', 20)} {pad('结果', 15)} {pad('功耗', 7)} {pad('SM频率', 9)} {pad('温度', 5)} 说明")
    for key, entry in results.items():
        if "skipped" in entry:
            continue
        power, clock, temp = entry["power"], entry["clock"], entry["temp"]
        note = "对照项，解锁前后应基本不变" if entry["kind"] == "ctrl" else "受限项，解锁后应变快"
        ceiling = locked_ceiling(key, info, clock)
        if ceiling:
            note = f"锁定上限 {ceiling:.1f}，实测为其 {entry['value'] / ceiling:.0%}"
        print(f"{pad(entry['name'], 20)} {pad(format_value(entry), 15)} "
              f"{pad(f'{power:.0f} W' if power else '-', 7)} {pad(f'{clock:.0f} MHz' if clock else '-', 9)} "
              f"{pad(f'{temp:.0f}°C' if temp else '-', 5)} {note}")
    limit = info.get("power_limit")
    if limit and any(e.get("power") and e["power"] > 0.95 * limit for e in results.values()):
        print(f"（有测试顶到了 {limit:.0f} W 功耗墙：这时卡在降频，解锁的收益会被功耗墙吃掉一部分）")


def print_verdict(data):
    info, results = data["info"], data["results"]
    for key in ("bf16", "fp16"):
        entry = results.get(key, {})
        ceiling = locked_ceiling(key, info, entry.get("clock"))
        if "value" not in entry or not ceiling:
            continue
        ratio = entry["value"] / ceiling
        if ratio > 1.10:
            verdict = "超过了锁定状态的理论上限 → 这张卡现在确实是解锁状态"
        elif ratio < 1.0:
            verdict = "没超过锁定状态的理论上限 → 看起来仍是锁定（半速）状态"
        else:
            verdict = "刚好在锁定上限附近，无法单次判断，以前后对比为准"
        print(f"\n判断：{entry['name']} 实测 {entry['value']:.1f} TFLOPS，是锁定上限 {ceiling:.1f} 的 {ratio:.0%}，{verdict}。")
        return
    if info.get("sm"):
        print("\n判断：没采到 SM 频率（nvidia-smi 不可用？），只能靠前后对比。")


def compare(before_path, after_path):
    before, after = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (before_path, after_path))
    for label, run_ in (("解锁前", before), ("解锁后", after)):
        print(f"{label} ({run_['info']['time']}): {unlock_state(run_['info']['kernel_log'])}")
    for key, label in (("gpu", "GPU"), ("uuid", "GPU UUID"), ("driver", "驱动版本"), ("torch", "PyTorch"),
                       ("cuda", "CUDA"), ("power_limit", "功耗墙")):
        if before["info"].get(key) != after["info"].get(key):
            print(f"注意：两次的{label}不同（{before['info'].get(key)} → {after['info'].get(key)}），对比不严格")
    if before["size"] != after["size"]:
        print(f"注意：两次的矩阵尺寸不同（{before['size']} → {after['size']}），对比不严格")

    print(f"\n{pad('测试', 20)} {pad('解锁前', 9)} {pad('解锁后', 9)} {pad('变化', 8)} {pad('输出', 9)} 结论")
    mismatched = []
    for key, b in before["results"].items():
        a = after["results"].get(key, {})
        if "value" not in a or "value" not in b:
            print(f"{pad(b['name'], 20)} 跳过")
            continue
        change = a["value"] / b["value"] - 1
        same = "逐位一致" if a["hash"] == b["hash"] else "不同！"
        if a["hash"] != b["hash"]:
            mismatched.append(b["name"])
        if b["kind"] == "ctrl":
            verdict = "正常（对照项没变）" if abs(change) <= 0.05 else "对照项也变了，两次条件可能不一致（温度/功耗墙/后台程序）"
        elif change >= 0.15:
            verdict = "变快了，解锁生效"
        elif abs(change) < 0.05:
            verdict = "没变化，解锁没生效"
        elif change < 0:
            verdict = "变慢了，两次条件可能不一致（温度/功耗墙/后台程序）"
        else:
            verdict = "提升很小，可能被功耗墙/温度限制，或解锁未完全生效"
        print(f"{pad(b['name'], 20)} {b['value']:9.1f} {a['value']:9.1f} {change:+8.0%} {pad(same, 9)} {verdict}")
    print("\n单位：INT8 为 TOPS，其余为 TFLOPS。「输出」比较的是固定输入下的计算结果哈希，"
          "gpu-unlock 只改速度，同一套驱动/PyTorch 下应逐位一致。")
    if mismatched:
        same_setup = before["size"] == after["size"] and all(
            before["info"].get(k) == after["info"].get(k) for k in ("driver", "torch", "cuda", "gpu"))
        print(f"输出不一致：{'、'.join(mismatched)}。"
              + ("GPU、驱动、PyTorch、矩阵尺寸都相同却不一致，这不该发生，建议带上两个 JSON 去 gpu-unlock 的 issue 反馈。"
                 if same_setup else "两次的 GPU/驱动/PyTorch/矩阵尺寸不同，哈希不同是正常的，换成同样的条件再比。"))


def main():
    parser = argparse.ArgumentParser(description="gpu-unlock 解锁前后对比跑分（FP16/BF16/TF32 张量核心吞吐）")
    parser.add_argument("--out", help="结果 JSON 的保存路径（默认按时间命名）")
    parser.add_argument("--size", type=int, default=8192, help="方阵边长，默认 8192；显存小的卡可用 4096")
    parser.add_argument("--seconds", type=float, default=3.0, help="每项测量时长，默认 3 秒")
    parser.add_argument("--device", default="cuda", help="默认 cuda（第一张卡），可写 cuda:1；cpu 只用来试跑脚本")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="对比两次保存的 JSON")
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
