import csv

# Archivo de entrada (texto original)
input_file = "mesh.txt"
# Archivo de salida CSV
output_file = "nodes.csv"

data_started = False
rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Detectar inicio (línea que contiene 'Triangle')
        if "Triangle" in line:
            data_started = True
            continue

        # Detectar fin
        if "end coordinates" in line.lower():
            break

        if data_started:
            parts = line.split()
            # Solo líneas con 4 valores numéricos: id x y z
            if len(parts) == 4:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    rows.append([node_id, x, y, z])
                except ValueError:
                    pass  # Ignorar líneas no numéricas

# Escribir el CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["node_id", "x", "y", "z"])
    writer.writerows(rows)

print(f"CSV creado correctamente: {output_file}")
