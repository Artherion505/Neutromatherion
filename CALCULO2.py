import numpy as np
import matplotlib.pyplot as plt

# 1. PARÁMETROS CALIBRADOS
G = 4.300e-6          # Constante Gravitacional [(kpc) * (km/s)^2 / M_sun]
M_disk = 1.2e10       # Masa del disco bariónico [M_sun]
R_disk = 2.5          # Radio de escala del disco [kpc]

# Parámetros del Campo Neutromatherion N(r)
g_coupling = 3.8e-3   # Constante de acoplamiento escalar
m_N = 0.005           # Masa efectiva del campo (rango extendido)

# Dominio de radio galáctico
r = np.linspace(0.1, 25.0, 500)

# 2. PERFIL DE MASA BARIÓNICA Y VELOCIDADES
M_b = M_disk * (1 - (1 + r / R_disk) * np.exp(-r / R_disk))
v_newton = np.sqrt(G * M_b / r)

term_reactio = (g_coupling**2 * M_b / (4 * np.pi)) * ((1 / r) + m_N) * np.exp(-m_N * r)
v_unitario = np.sqrt(v_newton**2 + term_reactio)

# 3. PUNTOS SINTETICOS DE EJEMPLO (Inspirados en radios SPARC)
np.random.seed(42)
r_sparc = np.array([0.8, 1.8, 3.2, 5.0, 7.5, 10.0, 13.0, 16.0, 19.0, 22.0, 24.5])
M_b_sparc = M_disk * (1 - (1 + r_sparc / R_disk) * np.exp(-r_sparc / R_disk))

term_reactio_sparc = (g_coupling**2 * M_b_sparc / (4 * np.pi)) * ((1 / r_sparc) + m_N) * np.exp(-m_N * r_sparc)
v_sparc_true = np.sqrt((G * M_b_sparc / r_sparc) + term_reactio_sparc)

v_sparc_obs = v_sparc_true + np.random.normal(0, 2.5, size=len(r_sparc))
v_err = np.random.uniform(3.5, 5.5, size=len(r_sparc))

# 4. GRAFICADO Y EXPORTACIÓN
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.errorbar(r_sparc, v_sparc_obs, yerr=v_err, fmt='o', color='#d62728',
            ecolor='#ff9896', elinewidth=1.8, capsize=4, capthick=1.5,
            label='Datos sintéticos de ejemplo', zorder=5)

ax.plot(r, v_newton, linestyle='--', color='#1f77b4', linewidth=2.0,
        label=r'Newtoniano Bariónico ($v \propto r^{-1/2}$)')

ax.plot(r, v_unitario, linestyle='-', color='#2ca02c', linewidth=2.5,
        label=r'Modelo $N(r)$ (Tejido Reactio del Neutromatherion)')

ax.fill_between(r, v_newton, v_unitario, color='#2ca02c', alpha=0.18,
                label=r'Aporte de la Fuerza Reactiva $g \cdot r \frac{dN}{dr}$')

ax.set_title('Curva de Rotación Galáctica: Física Estándar vs. Cosmología de la Unidad', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Radio Galáctico $r$ [kpc]', fontsize=11)
ax.set_ylabel('Velocidad Orbital $v(r)$ [km/s]', fontsize=11)
ax.set_xlim(0, 25)
ax.set_ylim(0, 110)

ax.annotate('Región Asintótica / Plana\n' + r'$v_{flat} \approx \sqrt{\frac{g^2 M_b}{4\pi}}$',
            xy=(18, v_unitario[360]), xytext=(12, 85),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", lw=0.8))

ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
plt.tight_layout()

# Guardar figura para LaTeX
plt.savefig('curva_rotacion_unidad.png', dpi=300)
plt.show()