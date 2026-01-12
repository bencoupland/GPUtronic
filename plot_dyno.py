# plot_dyno.py - Simple plot of dyno CSV
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gputronic_master_dyno.csv")
df = df[df['column'] == 'DATA']  # filter sweep data

plt.figure(figsize=(10,6))
plt.plot(df['sweep_tgt'], df['f_actual'], label='Actual RPM')
plt.plot(df['sweep_tgt'], df['sweep_tgt'], '--', label='Target RPM')
plt.xlabel('Target RPM')
plt.ylabel('Actual RPM')
plt.title('GPUtronic Dyno Sweep')
plt.legend()
plt.grid(True)
plt.savefig('dyno_plot.png')
plt.show()
