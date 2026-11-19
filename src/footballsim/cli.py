"""Command line interface: footballsim download | train | evaluate | simulate | app."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__
from . import config as C


def _cmd_download(args):
    from .data import download_events

    comps = [tuple(c.split(":", 1)) for c in args.competition] if args.competition else None
    ev = download_events(comps, force=args.force)
    print(f"{len(ev):,} events from {ev.match_id.nunique()} matches -> {C.events_path()}")


def _load_seqs():
    import pandas as pd

    from .data import build_possessions

    if not C.events_path().exists():
        sys.exit("No data found. Run `footballsim download` first.")
    return build_possessions(pd.read_parquet(C.events_path()))


def _cmd_train(args):
    from .train import save_checkpoint, train_model

    seqs = _load_seqs()
    print(f"{len(seqs):,} possessions from {seqs.match_id.nunique()} matches")
    overrides = {}
    if args.d_model:
        overrides["d_model"] = args.d_model
    if args.layers:
        overrides["n_layers"] = args.layers
    ckpt = train_model(seqs, epochs=args.epochs, patience=args.patience, lr=args.lr,
                       seed=args.seed, cfg_overrides=overrides)
    path = save_checkpoint(ckpt)
    print(f"saved {path}")


def _cmd_evaluate(args):
    from .evaluate import evaluate, save_report
    from .plotting import hero_figure
    from .simulate import simulate, zone_distribution
    from .train import device_auto, load_checkpoint

    device = device_auto()
    model, vocab, ckpt = load_checkpoint(device=device)
    seqs = _load_seqs()
    report = evaluate(model, vocab, ckpt, seqs, n_sim=args.n_sim, device=device)
    path = save_report(report)

    te = seqs[seqs.match_id.isin(ckpt["split"]["test"])].toks.tolist()
    sim = simulate(model, vocab, args.n_sim, seed=0, device=device)
    fig = hero_figure(report, zone_distribution(te), zone_distribution(sim))
    out = Path(args.figure)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")

    print(f"Markov perplexity       : {report['markov_perplexity']:.2f}")
    print(f"Transformer perplexity  : {report['transformer_perplexity']:.2f} "
          f"({report['improvement_pct']:.1f}% lower)")
    print(f"Shot calibration gap    : {report['shot_calibration_gap']:.4f}")
    print("real      :", {k: round(v, 4) for k, v in report["real"].items()})
    print("simulated :", {k: round(v, 4) for k, v in report["simulated"].items()})
    print(f"report -> {path} | figure -> {out}")


def _cmd_simulate(args):
    from .data import make_token
    from .simulate import simulate, summarize
    from .train import load_checkpoint

    model, vocab, _ = load_checkpoint()
    prefix = []
    if args.start_zone is not None:
        prefix = [make_token(args.start_action, args.start_outcome, args.start_zone)]
    sims = simulate(model, vocab, args.n, prefix=prefix, temperature=args.temperature, seed=args.seed)
    print({k: round(v, 4) for k, v in summarize(sims).items()})
    for s in sims[: args.show]:
        print(" -> ".join(s))


def _cmd_app(args):
    app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    if not app.exists():
        sys.exit(f"App file not found at {app}. Run from a source checkout (pip install -e .).")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="footballsim", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download StatsBomb open data")
    d.add_argument("--competition", action="append",
                   help="'Name:Season', repeatable. Default: FIFA World Cup 2018 and 2022. "
                        "Example: --competition 'La Liga:2018/2019'")
    d.add_argument("--force", action="store_true", help="re-download even if cached")
    d.set_defaults(fn=_cmd_download)

    t = sub.add_parser("train", help="train the transformer")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--d-model", type=int, default=None)
    t.add_argument("--layers", type=int, default=None)
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("evaluate", help="evaluate on held-out matches and write report + figure")
    e.add_argument("--n-sim", type=int, default=10000)
    e.add_argument("--figure", default=str(Path("docs") / "hero.png"))
    e.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("simulate", help="simulate possessions from a starting state")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--start-zone", type=int, default=None, help=f"0..{C.N_ZONES - 1}")
    s.add_argument("--start-action", default="Pass")
    s.add_argument("--start-outcome", default="ok")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--show", type=int, default=5, help="print this many example possessions")
    s.set_defaults(fn=_cmd_simulate)

    a = sub.add_parser("app", help="launch the interactive web app")
    a.set_defaults(fn=_cmd_app)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
