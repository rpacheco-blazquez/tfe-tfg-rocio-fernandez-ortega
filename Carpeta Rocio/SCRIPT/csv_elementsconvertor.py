import csv
import os

def extraer_elementos_a_csv(archivo_entrada, archivo_salida="elementss.csv"):
    """
    Extrae información de elementos desde un archivo .flavia y los guarda en CSV.
    Procesa datos entre 'Elements' y 'End Elements'.
    
    Args:
        archivo_entrada: C:/Users/rocio/OneDrive/Desktop/numericalTFGsegundo.gid/numericalTFGsegundo.flavia
        archivo_salida: elementss.csv
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

            # Detectar inicio (línea que contiene 'Elements')
            if "elements" in line.lower():
                data_started = True
                continue

            # Detectar fin
            if "end elements" in line.lower():
                break

            if data_started:
                parts = line.split()
                # Formato esperado: element_id node_1 node_2 node_3 (4 valores)
                if len(parts) == 4:
                    try:
                        elem_id = int(parts[0])  # ID del elemento
                        n1 = int(parts[1])        # Nodo 1
                        n2 = int(parts[2])        # Nodo 2
                        n3 = int(parts[3])        # Nodo 3
                        rows.append([elem_id, n1, n2, n3])
                    except ValueError:
                        pass  # Ignora líneas con formato incorrecto

    # Verificar que se encontraron datos
    if not rows:
        print("⚠️ Advertencia: No se encontraron elementos en el archivo.")
        print("   Verifica que el archivo contiene las secciones 'Elements' y 'End Elements'.")
        return

    # Escribir el CSV
    with open(archivo_salida, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Encabezado del CSV
        writer.writerow(["element_id", "node_1", "node_2", "node_3"])
        writer.writerows(rows)

    print(f"✅ CSV creado correctamente: {archivo_salida}")
    print(f"📊 Total de elementos procesados: {len(rows)}")


if __name__ == "__main__":
    # ========================================
    # CONFIGURA AQUÍ TUS RUTAS DE ARCHIVO
    # ========================================
    
    # OPCIÓN 1: Especificar rutas directamente
    input_file = r"C:\Users\rocio\OneDrive\Desktop\numericalTFGsegundo.gid\numericalTFGsegundo.flavia"
    output_file = r"C:\Users\rocio\Repositorio TFG Visual Code\tfe-tfg-rocio-fernandez-ortega\Carpeta Rocio\SCRIPT\elementss.csv"
    
    # OPCIÓN 2: Pedir al usuario (descomenta estas líneas para activar)
    # print("🔍 Extractor de elementos desde archivo .flavia")
    # print("-" * 50)
    # input_file = input("Ingresa la ruta completa del archivo de entrada (.txt/.flavia): ").strip('"')
    # output_file = input("Ingresa la ruta completa del archivo CSV de salida (o presiona Enter para 'elements.csv'): ").strip('"')
    # if not output_file:
    #     output_file = "elementss.csv"
    
    # OPCIÓN 3: Usar ruta completa del ordenador
    # input_file = "C:/Users/rocio/OneDrive/Desktop/numericalTFGsegundo.gid/numericalTFGsegundo.flavia"
    # output_file = "C:\\Users\\rocio\\MiCarpeta\\datos\\elementss.csv"
    
    print(f"\n📂 Archivo de entrada: {input_file}")
    print(f"💾 Archivo de salida: {output_file}\n")
    
    extraer_elementos_a_csv(input_file, output_file)