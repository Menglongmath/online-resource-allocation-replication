"""Tables 1-3, Figure 1 and every number quoted in Section 6 (and the introduction) of the paper.

Reads the package's result files and the small-network instances:
  results/small_network/runs.csv, runs_vera_banerjee.csv   (code/small/run_small.py, run_vb.py)
  results/dmd_step_check/total_reward.csv                  (code/small/run_dmd_step_check.py)
  results/huge/runs.csv, instance.json, upper_bound_validation.json   (code/huge)
  data/small_network/instances                             (instance sizes quoted in Section 6.1)
Writes results/paper/{table_known,table_unknown,table_huge}.tex, fig_regret.{pdf,png} and numbers.txt.
Usage: python build_paper_tables.py
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = RES / 'paper'
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('OLP_SCRATCH', Path.home() / '.olp_scratch')) / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = [1000, 5000, 10000, 20000, 50000, 100000]
HT = [1000, 5000, 10000, 100000]
KNOWN = ['ours_known', 'ours_known_lp_start', 'vera_banerjee', 'he_fallback']
UNKNOWN = ['ours_estimated', 'dmd_fallback', 'ma2025_fallback']
NAME = {'ours_known': 'Our policy', 'ours_known_lp_start': 'Our policy, LP start',
        'vera_banerjee': r'\citet{vera_banerjee_2021}', 'he_fallback': r'\citet{he_wei_xu_yu_2025}',
        'ours_estimated': 'Our policy', 'dmd_fallback': 'DMD', 'ma2025_fallback': r'\citet{ma_cao_tsang_xia_2025}'}
LABEL = {'ours_known': 'Our policy', 'ours_known_lp_start': 'Our policy, LP start',
         'vera_banerjee': 'Vera and Banerjee (2021)', 'he_fallback': 'He et al. (2025)', 'ours_estimated': 'Our policy',
         'dmd_fallback': 'DMD', 'ma2025_fallback': 'Ma et al. (2025)'}


def load_small():
    d = pd.read_csv(RES / 'small_network' / 'runs.csv')
    vb = pd.read_csv(RES / 'small_network' / 'runs_vera_banerjee.csv')
    d = pd.concat([d, vb[d.columns.intersection(vb.columns)]], ignore_index=True)
    assert not d.duplicated(['policy', 'case', 'horizon', 'path']).any()
    assert d.groupby(['policy', 'case', 'horizon']).size().eq(30).all()
    ref = d[d.policy == 'ours_known'].set_index(['case', 'horizon', 'path']).lp_hindsight
    x = d.set_index(['case', 'horizon', 'path']).lp_hindsight
    assert np.allclose(x.values, ref.loc[x.index].values, rtol=1e-12)
    return d


def table(inst, rows, label, caption, tabcolsep):
    t = {p: {h: (inst.loc[p][h].mean(), inst.loc[p][h].min(), inst.loc[p][h].max()) for h in HT} for p in rows}
    best = {h: min(rows, key=lambda p: t[p][h][0]) for h in HT}
    L = [r'\begin{table}[!htb]\centering', r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\begingroup\small\setlength{\tabcolsep}{' + tabcolsep + '}',
         r'\begin{tabular*}{\linewidth}{@{}l@{\hspace{6pt}\extracolsep{\fill}}*{12}{r}@{}}\toprule',
         ' & ' + ' & '.join(r'\multicolumn{3}{c}{$T=' + f'{h:,}'.replace(',', '{,}') + '$}' for h in HT) + r' \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
         'Policy & ' + ' & '.join(['Mean & Min & Max'] * 4) + r' \\\midrule']
    for p in rows:
        cells = []
        for h in HT:
            m, lo, hi = t[p][h]
            cells += [(r'\textbf{%.3f}' if best[h] == p else '%.3f') % m, '%.3f' % lo, '%.3f' % hi]
        L.append(NAME[p] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule\end{tabular*}', r'\par\vspace{3pt}\raggedright\footnotesize',
          'Mean, minimum, and maximum over the 30 instances of the instance-average percentage loss; each instance '
          'average is taken over the 30 demand sequences. The lowest mean at each horizon is in bold.',
          r'\par\endgroup\end{table}']
    (OUT / ('table_' + label.split('numerical-')[1] + '.tex')).write_text('\n'.join(L) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_small()
    inst = d.groupby(['policy', 'case', 'horizon'])[['percentage_loss', 'regret']].mean()
    loss = inst.percentage_loss.unstack('horizon')
    mean = loss.groupby(level='policy').mean()[H]
    table(loss, KNOWN, 'tab:numerical-known', 'Known arrival probabilities. Percentage loss relative to the hindsight LP.',
          '2.4pt')
    table(loss, UNKNOWN, 'tab:numerical-unknown',
          'Unknown arrival probabilities. Percentage loss relative to the hindsight LP.', '3pt')
    lines = []
    P = lambda *a: lines.append(' '.join(str(v) for v in a))
    P('Section 6.1: instance sizes over the 30 instances and 6 horizons')
    sizes = []
    for case in sorted(d.case.unique()):
        for h in H:
            folder = ROOT / 'data' / 'small_network' / 'instances' / case / str(h)
            with np.load(folder / 'model.npz') as z:
                tp = z['type_ptr']
            with np.load(folder / 'inventory.npz') as z:
                stocked = int((z['capacity'] > 0).sum())
            sizes.append((len(tp) - 1, stocked, int(tp[-1]), int(np.diff(tp).max())))
    sizes = np.array(sizes)
    P(f'  request types {sizes[:, 0].min()}-{sizes[:, 0].max()}; item-center pairs with positive inventory '
      f'{sizes[:, 1].min()}-{sizes[:, 1].max()}; actions other than rejection {sizes[:, 2].min()}-{sizes[:, 2].max()}, '
      f'at most {sizes[:, 3].max()} for one type')
    P('\nMean percentage loss over the 30 instances (each averaged over its 30 sequences), all six horizons')
    P(mean.loc[KNOWN + UNKNOWN].round(3).to_string())
    x = loss.stack().unstack('policy')
    def wins(a, b):
        return (x[a] < x[b]).groupby(level='horizon').sum().reindex(H).tolist()
    P('\nInstances (of 30) in which the first policy has the lower instance-average loss, horizons', H)
    for a, b in [('vera_banerjee', 'ours_known_lp_start'), ('ours_known_lp_start', 'he_fallback'),
                 ('ours_known', 'he_fallback'), ('ours_estimated', 'dmd_fallback'), ('ours_estimated', 'ma2025_fallback')]:
        P(f'  {a} vs {b}:', wins(a, b))
    P('\nOriginal versions without the fallback rule: mean percentage loss, and whether it is higher than with the rule')
    for p in ['he', 'dmd', 'ma2025']:
        P(f'  {p}:', mean.loc[p].round(3).tolist(), ' higher at every horizon:',
          bool((mean.loc[p] > mean.loc[p + '_fallback']).all()))
    rank = inst.regret.xs(100000, level='horizon').unstack('policy')
    P('\nFigure 1: rank (1 = smallest) of instance I40_M3 among the 30 instances by regret at T = 100,000')
    P(' ', {p: int(rank[p].rank()['I40_M3']) for p in KNOWN + UNKNOWN})
    st = pd.read_csv(RES / 'dmd_step_check' / 'total_reward.csv', index_col=0)
    P('\nSection 6.2, DMD step sizes on 30 other instances: total reward'); P(st.round(1).to_string())
    best = st.drop('paper').sum(axis=1).idxmax()
    P(f'  best grid value {best}; it earns less than T^(-1/2) at every horizon:', bool((st.loc[best] < st.loc['paper']).all()))

    # Figure 1: regret for the instance (n, M) = (40, 3), averaged over its 30 sequences
    R = inst.regret.unstack('horizon')[H]
    col = {'ours_known': '#2a78d6', 'ours_known_lp_start': '#2a78d6', 'ours_estimated': '#2a78d6',
           'vera_banerjee': '#1a1a1a', 'he_fallback': '#8c5a2b', 'dmd_fallback': '#4a3aa7', 'ma2025_fallback': '#e34948'}
    mk = {'ours_known': 'o', 'ours_known_lp_start': 'o', 'ours_estimated': 'o', 'vera_banerjee': 'D',
          'he_fallback': 's', 'dmd_fallback': 'v', 'ma2025_fallback': 'X'}
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#8a8984', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    xs = np.array(H, float)
    for ax, (title, rows, top) in zip(axes, [('Known arrival probabilities', KNOWN, 120),
                                             ('Unknown arrival probabilities', UNKNOWN, 300)]):
        notes = []
        for p in rows:
            y = R.loc[(p, 'I40_M3')].values
            ax.plot(xs, y, color=col[p], marker=mk[p], ls='--' if 'lp_start' in p else '-', ms=5,
                    lw=2.4 if p.startswith('ours') else 1.5, label=LABEL[p], markeredgecolor='white',
                    markeredgewidth=0.6, zorder=4 if p.startswith('ours') else 3)
            off = [f'{int(round(v)):,} at {int(h / 1000)}k' for h, v in zip(H, y) if v > top]
            if off:
                notes.append(LABEL[p] + ' off scale:\n' + ', '.join(off))
        if notes:
            ax.annotate('\n'.join(notes), xy=(0.02, 0.98), xycoords='axes fraction', fontsize=6.3, color='#52514e',
                        ha='left', va='top', zorder=6, bbox=dict(facecolor='white', edgecolor='none', alpha=0.92, pad=1.5))
        ax.set_xscale('log'); ax.set_xticks(H); ax.set_xticklabels(['1k', '5k', '10k', '20k', '50k', '100k'])
        ax.minorticks_off(); ax.set_xlim(8.5e2, 1.15e5); ax.set_ylim(0, top)
        ax.grid(True, color='#e9e8e4', lw=0.6, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=9); ax.set_xlabel('Horizon $T$ (log scale)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.2)
    axes[0].set_ylabel('Regret'); fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_regret.{ext}', dpi=200, bbox_inches='tight')

    # Table 3 (huge instance)
    hu = pd.read_csv(RES / 'huge' / 'runs.csv')
    HN = {'ours_estimated': 'Our policy', 'dmd_fallback': 'DMD', 'ma2025_fallback': r'\citet{ma_cao_tsang_xia_2025}'}
    HH = [10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]
    g = hu.groupby(['policy', 'horizon']).loss_percent.agg(['mean', 'min', 'max'])
    hu['us'] = 1e6 * hu.cpu_seconds / hu.horizon
    rng = max(float((g.loc[p]['max'] - g.loc[p]['min']).max()) for p in HN)
    rows = []
    for p in HN:
        best = [min(HN, key=lambda q: g.loc[(q, h), 'mean']) for h in HH]
        vals = [(r'\textbf{%.2f}' if best[i] == p else '%.2f') % g.loc[(p, h), 'mean'] for i, h in enumerate(HH)]
        us = hu[hu.policy == p].us
        rows.append(HN[p] + ' & ' + ' & '.join(vals) + ' & %.0f--%.0f' % (round(us.min()), round(us.max())) + r' \\')
    (OUT / 'table_huge.tex').write_text('\n'.join([
        r'\begin{table}[!htb]\centering',
        r'\caption{Instance with $9.98\times10^{10}$ type--action pairs, unknown arrival probabilities. Percentage loss relative to an upper bound on the hindsight LP, and CPU time per request.}',
        r'\label{tab:numerical-huge}', r'\begingroup\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}\toprule',
        r' & \multicolumn{4}{c}{Percentage loss} & CPU time \\', r'\cmidrule(lr){2-5}',
        r'Policy & $T=10^{5}$ & $T=10^{6}$ & $T=10^{7}$ & $T=10^{8}$ & ($\mu$s per request) \\\midrule', *rows,
        r'\bottomrule\end{tabular*}', r'\par\vspace{3pt}\raggedright\footnotesize',
        r'Mean over demand sequences (10, 10, 10, 5 sequences at the 4 horizons) of the percentage loss $100\,(U-\REV^\mu_T)/U$, where $U\ge\Phi(B_T,N)$ is the upper bound in the text; the largest range over sequences of any entry is '
        + '%.2f' % rng + r' percentage points. CPU time is the range, over all runs, of the CPU time of a run divided by $T$. The lowest loss at each horizon is in bold.',
        r'\par\endgroup\end{table}']) + '\n')
    P('\nSection 6.4: instance statistics by horizon'); P(json.loads((RES / 'huge' / 'instance.json').read_text()))
    val = json.loads((RES / 'huge' / 'upper_bound_validation.json').read_text())
    P('  U exceeds the hindsight LP on the smaller instance by (percent):', [round(v['excess_percent'], 4) for v in val])
    P('Mean percentage loss relative to U'); P(g['mean'].unstack('horizon').round(3).to_string())
    P('CPU microseconds per request (min, max):', hu.groupby('policy').us.agg(['min', 'max']).round(2).to_dict('index'))
    (OUT / 'numbers.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
