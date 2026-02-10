import re
import csv
from collections import defaultdict
import os

def extract_total_elevation_data(input_file, output_file):
    """
    Extrae datos de Total elevation del archivo .res y los guarda en CSV
    
    Args:
        input_file: C:/Users/rocio/OneDrive/Desktop/numericalTFGsegundo.gid/numericalTFGsegundo.flavia.res
        output_file: altura_malla.csv
    """
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(input_file):
        print(f"❌ Error: El archivo '{input_file}' no existe.")
        return
    
    # Diccionario para almacenar los datos: {time: {node_id: value}}
    data_by_time = {}
    all_nodes = set()
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        current_time = None
        in_total_elevation = False
        in_values = False
        
        for line in f:
            line = line.strip()
            
            # Detectar inicio de bloque Total elevation
            if 'Result "Total elevation' in line:
                # Extraer el tiempo del nombre del resultado
                match = re.search(r'"Free surface"\s+([\d.]+)', line)
                if match:
                    current_time = float(match.group(1))
                else:
                    current_time = 0.0  # Primer timestamp sin número
                
                in_total_elevation = True
                data_by_time[current_time] = {}
                continue
            
            # Detectar inicio de valores
            if in_total_elevation and line == 'Values':
                in_values = True
                continue
            
            # Detectar fin de valores
            if in_values and line == 'End Values':
                in_values = False
                in_total_elevation = False
                continue
            
            # Leer los valores (node_id value)
            if in_values and line:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        node_id = int(parts[0])
                        value = float(parts[1])
                        data_by_time[current_time][node_id] = value
                        all_nodes.add(node_id)
                    except ValueError:
                        continue
    
    # Verificar que se encontraron datos
    if not data_by_time:
        print("⚠️ Advertencia: No se encontraron datos de Total elevation en el archivo.")
        return
    
    # Ordenar tiempos
    sorted_times = sorted(data_by_time.keys())
    
    # 🔧 INCLUIR TODOS LOS NODOS: del 1 hasta el máximo nodo encontrado
    if all_nodes:
        max_node = max(all_nodes)
        min_node = min(all_nodes)
        all_nodes_range = list(range(min_node, max_node + 1))
    else:
        print("❌ Error: No se encontraron nodos.")
        return
    
    print(f"📍 Generando CSV con TODOS los nodos del {min_node} al {max_node}...")
    
    # Escribir CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir encabezado
        header = ['time'] + all_nodes_range
        writer.writerow(header)
        
        # Escribir datos
        for time in sorted_times:
            row = [time]
            for node in all_nodes_range:
                # Si el nodo no tiene datos en este timestep, poner 0.0
                value = data_by_time[time].get(node, 0.0)
                row.append(value)
            writer.writerow(row)
    
    print(f"✅ CSV generado exitosamente: {output_file}")
    print(f"📊 Total de timestamps: {len(sorted_times)}")
    print(f"📍 Nodos CON datos de elevación: {len(all_nodes)}")
    print(f"📍 Total de columnas generadas: {len(all_nodes_range)}")
    print(f"📍 Rango de nodos: {min_node} - {max_node}")
    print(f"⏱️  Rango de tiempo: {sorted_times[0]:.2f} - {sorted_times[-1]:.2f}")
    print(f"\n⚠️  Nota: {len(all_nodes_range) - len(all_nodes)} nodos sin datos tendrán valor 0.0")


def extract_altura_malla(input_file, output_file):
    """
    Wrapper en español para mantener compatibilidad con llamadas existentes.
    Llama a `extract_total_elevation_data`.
    """
    return extract_total_elevation_data(input_file, output_file)

if __name__ == "__main__":
    # ========================================
    # CONFIGURA AQUÍ TUS RUTAS DE ARCHIVO
    # ========================================
    
    input_file = "C:/Users/rocio/OneDrive/Desktop/numericalTFGsegundo.gid/numericalTFGsegundo.flavia.res"
    output_file = "C:\\Users\\rocio\\Repositorio TFG Visual Code\\tfe-tfg-rocio-fernandez-ortega\\Carpeta Rocio\\SCRIPT\\altura_malla.csv"
    
    print(f"\n📂 Archivo de entrada: {input_file}")
    print(f"💾 Archivo de salida: {output_file}\n")
    
    extract_altura_malla(input_file, output_file)