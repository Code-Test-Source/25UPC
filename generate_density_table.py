import csv
from pathlib import Path
import numpy as np

from plot_density import air_density


def generate_table(
    temps=None,
    hums=None,
    presses=None,
    h0=0.0,
    h=0.0,
    out_csv=None,
):
    """Generate a flat CSV table of density for combinations of T (°C), RH (%) and P (Pa).

    Defaults (reasonable assumptions):
    - temps: 10 values from 0 to 40 °C
    - hums: 6 values from 0 to 100 %
    - presses: 6 values from 80000 to 105000 Pa
    """
    if temps is None:
        temps = np.linspace(0, 40.0, 9)
    if hums is None:
        hums = np.linspace(0.0, 100.0, 6)
    if presses is None:
        presses = np.linspace(80000.0, 105000.0, 6)

    temps = np.asarray(temps)
    hums = np.asarray(hums)
    presses = np.asarray(presses)

    if out_csv is None:
        out_csv = Path(__file__).with_name('density_table.csv')

    rows = []
    # iterate in a stable order: pressure outer, temp middle, humidity inner
    for P in presses:
        for T in temps:
            for RH in hums:
                rho = air_density(T, RH, P, h0=h0, h=h)
                rows.append((float(T), float(RH), float(P), float(rho)))

    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['T_C', 'RH_pct', 'P_Pa', 'rho_kg_m3'])
        for r in rows:
            writer.writerow(r)

    return out_csv, len(rows)


if __name__ == '__main__':
    csv_path, n = generate_table()
    print(f'Wrote {n} rows to: {csv_path}')
