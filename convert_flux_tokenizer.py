from transformers import T5TokenizerFast

tok_dir = r".\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev\tokenizer_2"

print("[flux] loading tokenizer from:", tok_dir)
tok = T5TokenizerFast.from_pretrained(tok_dir, legacy=False)
tok.save_pretrained(tok_dir)
print("[flux] fast tokenizer saved into", tok_dir)
