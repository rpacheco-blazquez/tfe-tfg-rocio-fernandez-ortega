import csv
import os

def extraer_coordenadas_a_csv(archivo_entrada, archivo_salida="nodes.csv"):
    """
    Extrae coordenadas de nodos desde un archivo .flavia y los guarda en CSV.
    Procesa datos entre 'Coordinates' y 'End Coordinates'.
    
    Args:
        archivo_entrada: C:/Users/rocio/OneDrive/Desktop/numericalTFGsegundo.gid/numericalTFGsegundo.flavia
        archivo_salida: nodess.csv
    """
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(archivo_entrada):
        print(f"❌ Error: El archivo '{archivo_entrada}' no existe.")
        return
    
    data_started = False
    rows = []

    with open(archivo_entrada, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detectar inicio (línea que contiene 'Coordinates')
            if line.lower() == "coordinates":
                data_started = True
                continue

            # Detectar fin
            if "end coordinates" in line.lower():
                break

            if data_started:
                parts = line.split()
                # Solo líneas con 4 valores numéricos: node_id x y z
                if len(parts) == 4:
                    try:
                        node_id = int(parts[0])    # ID del nodo
                        x = float(parts[1])         # Coordenada X
                        y = float(parts[2])         # Coordenada Y
                        z = float(parts[3])         # Coordenada Z
                        rows.append([node_id, x, y, z])
                    except ValueError:
                        pass  # Ignorar líneas no numéricas

    # Verificar que se encontraron datos
    if not rows:
        print("⚠️ Advertencia: No se encontraron coordenadas en el archivo.")
        print("   Verifica que el archivo contiene las secciones 'Coordinates' y 'End Coordinates'.")
        return

    # Escribir el CSV
    with open(archivo_salida, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Encabezado del CSV
        writer.writerow(["node_id", "x", "y", "z"])
        writer.writerows(rows)

    print(f"✅ CSV creado correctamente: {archivo_salida}")
    print(f"📊 Total de nodos procesados: {len(rows)}")


if __name__ == "__main__":
    # ========================================
    # CONFIGURA AQUÍ TUS RUTAS DE ARCHIVO
    # ========================================
    
    # OPCIÓN 1: Especificar rutas directamente
    input_file = "numericalTFGsegundo.flavia"
    output_file = "nodes.csv"
    
    # OPCIÓN 2: Pedir al usuario (descomenta estas líneas para activar)
    # print("🔍 Extractor de coordenadas desde archivo .flavia")
    # print("-" * 50)
    # input_file = input("Ingresa la ruta completa del archivo .flavia: ").strip('"')
    # output_file = input("Ingresa la ruta completa del archivo CSV de salida (o presiona Enter para 'nodes.csv'): ").strip('"')
    # if not output_file:
    #     output_file = "nodes.csv"
    
    # OPCIÓN 3: Usar ruta completa del ordenador
    # input_file = "C:\\Users\\rocio\\Repositorio TFG Visual Code\\tfe-tfg-rocio-fernandez-ortega\\Carpeta Rocio\\SCRIPT\\numericalTFGsegundo.flavia"
    # output_file = "C:\\Users\\rocio\\Repositorio TFG Visual Code\\tfe-tfg-rocio-fernandez-ortega\\Carpeta Rocio\\SCRIPT\\nodess.csv"
    
    print(f"\n📂 Archivo de entrada: {input_file}")
    print(f"💾 Archivo de salida: {output_file}\n")
    
    extraer_coordenadas_a_csv(input_file, output_file)