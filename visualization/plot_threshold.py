import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})


font_size = 18


# Sample data
np.random.seed(0)
id_sessions = np.random.uniform(low=0.32, high=1.3, size=30)  # ID sessions in the range 0.5 to 0.987
ood_sessions = np.random.uniform(low=2.1, high=4.0, size=50)  # OOD sessions in the range 2.1 to 4.0

# Threshold T
T = 1.712  # Updated threshold

# Plotting
plt.figure(figsize=(10, 4))
circle_size = 500
alpha = 0.2
plt.scatter(id_sessions, np.zeros_like(id_sessions), color='blue', label='ID Sessions', alpha=alpha, s=circle_size)
plt.scatter(ood_sessions, np.zeros_like(ood_sessions), color='red', label='OOD Sessions', alpha=alpha, s=circle_size)

# Separation line
plt.axvline(x=T, color='green', linestyle='--', label=f'Threshold $\\tau$')

# Annotations
plt.text(T + 0.1, 0, f'$\\tau$', color='green', ha='center', va='center', fontsize=font_size+10)
plt.yticks([])
plt.xticks(fontsize=font_size)
plt.xlabel('Number line for threshold $\\Delta(\\mathcal{D}_i), \\quad \\forall \\mathcal{D}_i \\in \\mathcal{D}_{ID} | \\mathcal{D}_{OOD}$', fontsize=font_size)
plt.title(f'Distribution of ID and OOD smaples over threshold line', fontsize=font_size)
plt.legend(fontsize=font_size - 2, handletextpad=0.8, labelspacing=0.8)
plt.grid(True)
plt.xlim(0.0, 4.0)  # Limit the x-axis between 0.0 and 4.0

plt.tight_layout(pad=0)
plt.savefig('visualization/fig_threshold_plot.pdf')
plt.show()

