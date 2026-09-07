import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Matriz de confusión obtenida del modelo Random Forest para S7
cm = np.array([
    [208,  21,  17],
    [  8, 218,  19],
    [ 24,  56, 679]
])

# Etiquetas de las clases
class_names = ['Clase 1\n(Abrir)', 'Clase 2\n(Cerrar)', 'Clase 3\n(Rotar)']

# Crear figura
plt.figure(figsize=(8, 6))

# Crear heatmap con seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14})

# Añadir títulos y etiquetas
plt.title('Matriz de Confusión - Random Forest (Sujeto S7)', fontsize=16, pad=20)
plt.xlabel('Clase Predicha', fontsize=14, labelpad=10)
plt.ylabel('Clase Real (Ground Truth)', fontsize=14, labelpad=10)

# Ajustar márgenes
plt.tight_layout()

# Guardar la imagen
output_path = 'matriz_confusion.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Matriz de confusión guardada exitosamente en: {output_path}")
