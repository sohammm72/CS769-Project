from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    MBartForConditionalGeneration, MBart50TokenizerFast,
)

ROOT = Path(__file__).resolve().parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_opus():
    p = ROOT / "opus"
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForSeq2SeqLM.from_pretrained(p).to(DEVICE).eval()

    def run(sents):
        batch = tok(sents, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**batch, num_beams=5, max_new_tokens=256)
        return tok.batch_decode(out, skip_special_tokens=True)
    return run


def load_nllb():
    p = ROOT / "nllb"
    tok = AutoTokenizer.from_pretrained(p, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(p).to(DEVICE).eval()

    def run(sents):
        batch = tok(sents, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **batch,
                forced_bos_token_id=tok.convert_tokens_to_ids("hin_Deva"),
                num_beams=5, max_new_tokens=256,
            )
        return tok.batch_decode(out, skip_special_tokens=True)
    return run


def load_mbart50():
    p = ROOT / "mbart50"
    tok = MBart50TokenizerFast.from_pretrained(p, src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained(p).to(DEVICE).eval()

    def run(sents):
        batch = tok(sents, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **batch,
                forced_bos_token_id=tok.convert_tokens_to_ids("hi_IN"),
                num_beams=5, max_new_tokens=256,
            )
        return tok.batch_decode(out, skip_special_tokens=True)
    return run


def load_indictrans2():
    from IndicTransToolkit.processor import IndicProcessor
    p = ROOT / "indictrans2"
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(p, trust_remote_code=True).to(DEVICE).eval()
    ip = IndicProcessor(inference=True)

    def run(sents):
        prepped = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="hin_Deva")
        inputs = tok(prepped, truncation=True, padding="longest",
                     return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, use_cache=False, min_length=0,
                                 max_length=256, num_beams=5, num_return_sequences=1)
        dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(dec, lang="hin_Deva")
    return run


def load_qwen():
    p = ROOT / "qwen"
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=DTYPE).to(DEVICE).eval()

    def run(sents):
        outs = []
        for s in sents:
            msgs = [
                {"role": "system", "content": "You are a professional English to Hindi translator. Output only the Hindi translation, nothing else."},
                {"role": "user", "content": s},
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                g = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            outs.append(tok.decode(g[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
        return outs
    return run


LOADERS = {
    "opus": load_opus,
    "nllb": load_nllb,
    "mbart50": load_mbart50,
    "indictrans2": load_indictrans2,
    "qwen": load_qwen,
}
