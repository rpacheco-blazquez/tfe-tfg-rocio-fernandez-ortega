import csv

# Archivo de entrada (texto original)
input_file = "E.txt"
# Archivo de salida
output_file = "elements.csv"

data_started = False
rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Inicio de los datos
        if line.lower() == "elements":
            data_started = True
            continue

        # Fin de los datos
        if line.lower() == "end elements":
            break

        if data_started:
            parts = line.split()
            # id_element + 3 nodos
            if len(parts) == 4:
                try:
                    elem_id = int(parts[0])
                    n1 = int(parts[1])
                    n2 = int(parts[2])
                    n3 = int(parts[3])
                    rows.append([elem_id, n1, n2, n3])
                except ValueError:
                    pass  # ignora líneas raras

# Escribir CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["element_id", "node_1", "node_2", "node_3"])
    writer.writerows(rows)

print(f"CSV creado correctamente: {output_file}")
