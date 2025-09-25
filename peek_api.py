import inspect
import wan.animate as W

print("\n== Module-level callables ==")
print([n for n in dir(W) if not n.startswith("_")])

print("\n== WanAnimate public attrs ==")
print([n for n in dir(W.WanAnimate) if not n.startswith("_")])

print("\n== WanAnimate method signatures ==")
for n in dir(W.WanAnimate):
    if n.startswith("_"): 
        continue
    obj = getattr(W.WanAnimate, n)
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            print(f"{n}{inspect.signature(obj)}")
        except Exception:
            print(f"{n} (no signature)")
