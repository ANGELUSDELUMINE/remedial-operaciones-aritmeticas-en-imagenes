import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar imágenes
img1 = cv2.imread('input1.png')
img2 = cv2.imread('input2.png')

if img1 is None or img2 is None:
    print("ERROR: No se encontraron las imágenes...")
    exit()

# Redimensionar si tienen tamaños distintos
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# ==========================================
# 2. Procesamiento y Guardado de Outputs individuales (OpenCV)
# ==========================================

# Suma Simple
added = cv2.add(img1, img2)
added_rgb = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
cv2.imwrite('output_suma_simple.png', added) # <-- Guarda la imagen limpia

# Suma Ponderada
weighted = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
weighted_rgb = cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB)
cv2.imwrite('output_suma_ponderada.png', weighted)

# Resta
subtracted = cv2.subtract(img1, img2)
subtracted_rgb = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)
cv2.imwrite('output_resta.png', subtracted)

# Bitwise AND
bitwise_and = cv2.bitwise_and(img1, img2)
and_rgb = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2RGB)
cv2.imwrite('output_bitwise_and.png', bitwise_and)

# Bitwise OR
bitwise_or = cv2.bitwise_or(img1, img2)
or_rgb = cv2.cvtColor(bitwise_or, cv2.COLOR_BGR2RGB)
cv2.imwrite('output_bitwise_or.png', bitwise_or)

# ==========================================
# 3. Visualización y Guardado de Reporte Completo (Matplotlib)
# ==========================================
plt.figure(figsize=(12, 8))

# Subplot 1: Imagen Original 1
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen 1 Original')
plt.axis('off') 

# Subplot 2: Suma Simple
plt.subplot(2, 3, 2)
plt.imshow(added_rgb)
plt.title('Suma Simple')
plt.axis('off')

# Subplot 3: Suma Ponderada
plt.subplot(2, 3, 3)
plt.imshow(weighted_rgb)
plt.title('Suma Ponderada (Mix)')
plt.axis('off')

# Subplot 4: Resta
plt.subplot(2, 3, 4)
plt.imshow(subtracted_rgb)
plt.title('Resta')
plt.axis('off')

# Subplot 5: Bitwise AND
plt.subplot(2, 3, 5)
plt.imshow(and_rgb)
plt.title('Bitwise AND')
plt.axis('off')

# Subplot 6: Bitwise OR
plt.subplot(2, 3, 6)
plt.imshow(or_rgb)
plt.title('Bitwise OR')
plt.axis('off')

plt.tight_layout()

# <-- GUARDA LA CUADRÍCULA COMPLETA DE 2x3 CON TÍTULOS Y EJES OCULTOS
plt.savefig('reporte_operaciones_aritmeticas.png', dpi=300, bbox_inches='tight')

# Muestra la ventana flotante en VS Code
plt.show()