
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone MATH JSONL evaluator for Qwen-style models.
"""
import argparse, os, re, json, time, datetime, torch, sympy
from sympy.parsing.latex import parse_latex
from modelscope import AutoModelForCausalLM
from transformers import AutoTokenizer

def extract_math_expression(text: str):
    if text is None:
        return ""
    try:
        start_index = text.rfind('\\boxed{')
        if start_index != -1:
            search_start = start_index + len('\\boxed{')
            brace_level = 1
            for i in range(search_start, len(text)):
                if text[i] == '{':
                    brace_level += 1
                elif text[i] == '}':
                    brace_level -= 1
                if brace_level == 0:
                    return text[search_start:i].strip()
    except Exception:
        pass
    paren_matches = re.findall(r'\((?:[^()]|\([^()]*\))*\)', text, flags=re.S)
    if paren_matches:
        return paren_matches[-1].strip()
    return text.strip()

def split_commas_outside_parens(s: str):
    parts, buf, level = [], "", 0
    for ch in s:
        if ch in "([{":
            level += 1; buf += ch
        elif ch in ")]}":
            level -= 1; buf += ch
        elif ch == "," and level == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    if buf.strip(): parts.append(buf)
    return parts

def extract_components(expr: str):
    if expr is None: return []
    expr = expr.strip().replace('\\left(', '(').replace('\\right)', ')').replace('\\left[', '[').replace('\\right]', ']')
    m = re.match(r'^[\(\[]\s*(.*)\s*[\)\]]$', expr, flags=re.S)
    if m:
        inner = m.group(1)
        return [p.strip() for p in split_commas_outside_parens(inner)]
    else:
        return [expr]

def try_parse_sympy(token: str):
    token = token.strip()
    try: return ("sympy", parse_latex(token))
    except Exception: pass
    try: return ("sympy", sympy.sympify(token))
    except Exception: pass
    try: return ("float", float(token))
    except Exception: return ("str", token)

def compare_tokenwise(a_str: str, b_str: str, tol: float = 1e-6) -> bool:
    a_kind, a_v = try_parse_sympy(a_str)
    b_kind, b_v = try_parse_sympy(b_str)
    if a_kind == "sympy" and b_kind == "sympy":
        try: return sympy.simplify(a_v - b_v) == 0
        except Exception:
            try: return abs(float(sympy.N(a_v, 50)) - float(sympy.N(b_v, 50))) <= tol
            except Exception: return False
    if a_kind == "sympy" and b_kind == "float":
        try:
            if abs(float(sympy.N(a_v, 50)) - b_v) <= tol: return True
            ns = sympy.nsimplify(b_v, [sympy.pi]); return sympy.simplify(ns - a_v) == 0
        except Exception: return False
    if a_kind == "float" and b_kind == "sympy":
        return compare_tokenwise(b_str, a_str, tol)
    if a_kind == "float" and b_kind == "float":
        return abs(a_v - b_v) <= tol
    def norm(s): return re.sub(r'\s+', '', s).rstrip('.,，')
    return norm(a_str) == norm(b_str)

def is_equivalent(expr1: str, expr2: str, tol: float = 1e-6) -> bool:
    if expr1 is None or expr2 is None: return False
    comps1 = extract_components(expr1); comps2 = extract_components(expr2)
    if len(comps1) != len(comps2): return False
    if not comps1: return True
    for a, b in zip(comps1, comps2):
        if not compare_tokenwise(a, b, tol=tol): return False
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="/gemini/code/V-Session/model/Qwen/Qwen2.5-3B")
    p.add_argument("--dataset_path", type=str, default="./data/MATH.jsonl")
    p.add_argument("--fewshot_prompt_path", type=str, default="./Prompt/math/V-Session_1-shot.txt")
    p.add_argument("--structured_instruction", type=str, default=r"Please reason step by step, and put your final answer within \boxed{}.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--do_sample", action="store_true")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path.rstrip("/"))
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    save_file_path = f"{args.save_dir}/{model_name}_{dataset_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}")
    print(f"Save file: {save_file_path}")

    fewshot_prompt = ""
    if os.path.exists(args.fewshot_prompt_path):
        with open(args.fewshot_prompt_path, "r", encoding="utf-8") as pf:
            fewshot_prompt = pf.read()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, device_map="auto")
    if getattr(model, "config", None) is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = []
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: obj = json.loads(line)
                except Exception: continue
                dataset.append(obj)
                if args.limit is not None and len(dataset) >= args.limit: break
    except FileNotFoundError:
        print(f"[Error] dataset not found: {args.dataset_path}")
        return

    acc_flags = []
    total_start_time = time.time()

    for idx, item in enumerate(dataset):
        each_start_time = time.time()
        question = item.get("problem") or item.get("question") or item.get("prompt") or item.get("input") or ""
        gold_answer_field = item.get("answer") or item.get("response") or item.get("label") or item.get("output")

        prompt = (args.structured_instruction or "") + question

        model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

        response, acc = "", False
        try:
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if args.do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
            else:
                gen_kwargs.update(dict(do_sample=False, temperature=0.0))

            with torch.no_grad():
                generated = model.generate(**gen_kwargs)
            gen_tokens = generated[0][input_ids.shape[1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            pred_expr = extract_math_expression(response)
            acc = is_equivalent(pred_expr, gold_answer_field)
        except Exception as e:
            print(f"[Warning] idx {idx} generation error: {e}")
            acc = False

        item['completion'] = response
        item['acc'] = bool(acc)
        acc_flags.append(1 if acc else 0)

        each_end_time = time.time()
        elapsed_time = each_end_time - each_start_time
        current_time = datetime.datetime.now().isoformat()

        log_entry = (
            f"[idx:{idx}] time:{current_time} elapsed:{elapsed_time:.2f}s acc:{acc}\n"
            f"Q: {question}\n"
            f"Gold: {gold_answer_field}\n"
            f"Pred: {extract_math_expression(response)}\n"
            f"Full completion: {response}\n\n"
        )
        print(log_entry)
        with open(save_file_path, "a", encoding="utf-8") as file:
            file.write(log_entry)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    correct_count = sum(acc_flags)
    total_count = len(acc_flags)
    mean_acc = (correct_count / total_count) if total_count > 0 else 0.0

    summary = (
        f"{'=' * 60}\n"
        f"Evaluation Finished\n"
        f"Total time: {total_time:.2f}s\n"
        f"Correct: {correct_count} / {total_count}\n"
        f"Accuracy: {mean_acc:.2%}\n"
        f"{'=' * 60}\n"
    )
    print(summary)
    with open(save_file_path, "a", encoding="utf-8") as file:
        file.write(summary)

if __name__ == "__main__":
    main()
