import glob, pandas as pd, numpy as np
PAIRS = {
  'baseline':  ('spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_baseline_*/audit_timeseries.csv',
                'TrajA_C2_30d_crew_0.100_20261105_080226/audit_timeseries.csv'),
  'Pmax_x0.80':('spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Pmax_x0.80_*/audit_timeseries.csv',
                'TrajA_C2_30d_crew_0.100_20261102_000819/audit_timeseries.csv'),
  'Pmax_x1.20':('spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Pmax_x1.20_*/audit_timeseries.csv',
                'TrajA_C2_30d_crew_0.100_20261103_145757/audit_timeseries.csv'),
  'Y_B_x0.80': ('spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Y_B_x0.80_*/audit_timeseries.csv',
                'TrajA_C2_30d_crew_0.100_20261103_071431/audit_timeseries.csv'),
}
for label,(p12,p17) in PAIRS.items():
    for res,p in [('12³',p12),('17³',p17)]:
        f = glob.glob(p)[0]
        df = pd.read_csv(f); t=df['time_h'].to_numpy(); c=df['CO2'].to_numpy()
        pk = c.max(); ipk = int(c.argmax())
        m99 = c >= 0.99*pk; m999 = c >= 0.999*pk
        print(f'{label:10s} {res}: peak={pk:.4e} @ {t[ipk]:.0f} h | '
              f'plateau99=[{t[m99].min():.0f},{t[m99].max():.0f}]h '
              f'w={t[m99].max()-t[m99].min():.0f}h | '
              f'plateau99.9=[{t[m999].min():.0f},{t[m999].max():.0f}]h')