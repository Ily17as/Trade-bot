from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"; RAW=DATA/"raw"; REF=DATA/"ref"; FEAT=DATA/"features"; CV=DATA/"cv"
for p in (DATA, RAW, REF, FEAT, CV): p.mkdir(parents=True, exist_ok=True)

def partdir(source,ticker,tf,y,m):
    p = RAW/source/ticker/tf/f"year={y:04d}"/f"month={m:02d}"; p.mkdir(parents=True, exist_ok=True); return p
