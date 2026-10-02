import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constantes físicas
c = 299792.458  # Velocidad de la luz (km/s)

# Parámetros extraídos del ajuste SPARC en tu modelo
# La velocidad se define como v^2 = v_barionica^2 + v_reactio^2
def v_cuadrado_barionico(r):
    # Aproximación del perfil kepleriano bariónico de tu gráfico
    return (180**2 * r) / (r**2 + 4)

def v_cuadrado_unidad(r):
    # Suma del componente bariónico y el campo reactivo N(r)
    return v_cuadrado_barionico(r) + 28500 * (1 - np.exp(-r/5))

# Aceleración gravitatoria efectiva g(r) = v^2 / r
def g_barionico(r):
    return v_cuadrado_barionico(r) / r

def g_unidad(r):
    return v_cuadrado_unidad(r) / r

# Integral de deflexión de luz (Aproximación de lente gravitacional débil)
# Ángulo = (4 / c^2) * Integral_b^inf [ g(r) * b / sqrt(r^2 - b^2) ] dr
def integrando_deflexion(r, b, g_func):
    return (g_func(r) * b) / np.sqrt(r**2 - b**2)

def calcular_deflexion(b, g_func):
    # La integración va desde el punto más cercano (b) hasta el infinito
    resultado, _ = quad(integrando_deflexion, b, np.inf, args=(b, g_func), limit=1000)
    return (4 / c**2) * resultado

# Generar un arreglo de radios para el parámetro de impacto b (en kpc)
radios_b = np.linspace(1, 25, 50)

# Calcular los ángulos de deflexión para ambos escenarios
deflexion_barionica = [calcular_deflexion(b, g_barionico) for b in radios_b]
deflexion_unidad = [calcular_deflexion(b, g_unidad) for b in radios_b]

# Convertir el resultado de radianes a arcosegundos (la medida astronómica estándar)
arcsec_barionico = np.array(deflexion_barionica) * 206265
arcsec_unidad = np.array(deflexion_unidad) * 206265

# Renderizado de la gráfica de resultados
plt.figure(figsize=(10, 6))
plt.plot(radios_b, arcsec_barionico, '--', label='Masa Bariónica (Física Estándar sin Materia Oscura)', color='blue')
plt.plot(radios_b, arcsec_unidad, '-', label='Modelo de la Unidad (Efecto del Neutromatherion)', color='red', linewidth=2)

plt.title("Predicción de Lente Gravitacional: Desviación de la Luz")
plt.xlabel("Distancia del rayo de luz al centro galáctico $b$ (kpc)")
plt.ylabel("Ángulo de deflexión (arcosegundos)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()