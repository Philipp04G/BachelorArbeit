import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# --- IMPORT DER MODULE ---
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData
from samson_data_reader.utils.gps_conversion import meter_to_gps_coords_old

# --- KONFIGURATION (PFADE ANPASSEN!) ---
# --- Pfade für Laptop
DATA_DIR = r"C:\Studium\BA_Projekt\DATEN"
MODEL_PATH = r"C:\Studium\BA_Projekt\BachelorArbeit\YOLO_Modell\baum_thesis\versuch_3_higherRes_Aug_300Epochs\weights\best.pt"
PATH_CALIB = r"C:\Studium\BA_Projekt\BachelorArbeit\SAMSON4_SAMSON3_stereo.yaml"

# --- Pfade für PC
DATA_DIR_PC = r"D:\Studium\BACHELOR ARBEIT\2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1"
MODEL_PATH_PC = r"D:\Studium\BACHELOR ARBEIT\YOLO_Modell\baum_thesis\versuch_3_higherRes_Aug_300Epochs\weights\best.pt"
PATH_CALIB_PC = r"D:\Studium\BACHELOR ARBEIT\BachelorArbeit\SAMSON4_SAMSON3_stereo.yaml"

# --- EINSTELLUNGEN ---
CSV_FILENAME = "gefundene_baeume.csv"
TARGET_CLASS = 'trunk' # <--- HIER DEN GENAUEN NAMEN AUS DEM DATENSATZ EINTRAGEN!
X_THRESHOLD = 150       # Pixel: Wie weit dürfen Segmente seitlich versetzt sein?
CONFIDENCE = 0.25      # YOLO Sicherheitsschwelle
ROTATE_IMAGE = True    # True, wenn Bilder um 90° gedreht werden müssen
MAX_DEPTH = 5.0       # Bäume weiter weg als 5m ignorieren
SKIP_IMAGES = 15       # Nur jedes X-te Bild verarbeiten

def filter_vertical_segments(boxes, x_thresh=150, overlap_thresh=0.3, pole_class_id=1):
    """
    Behält nur das unterste Segment eines Baumes (falls der Stamm zerteilt wurde).
    Prüft auch auf überlappende Boxen auf der X-Achse, um Mehrfacherkennungen 
    am selben dicken Stamm zu verhindern.
    """
    if boxes is None or len(boxes) == 0:
        return []

    candidates = []
    for i, box in enumerate(boxes):
        # 1. Klassen-ID der aktuellen Box auslesen
        class_id = int(box.cls[0].item())
        
        # 2. Ist es ein Pfosten? -> Sofort überspringen!
        if class_id == pole_class_id:
            continue

        b = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        
        center_x = (x1 + x2) / 2
        y_bottom = y2
        width = x2 - x1 # Breite der Box berechnen
        
        print(f"Baum gefunden auf X-Zentrum: {center_x:.1f}, Unten-Y: {y_bottom:.1f}")
        
        candidates.append({
            'id': i, 'center_x': center_x, 'y_bottom': y_bottom,
            'x1': x1, 'x2': x2, 'width': width
        })

    # Sortieren von links nach rechts
    candidates.sort(key=lambda c: c['center_x'])
    keep_indices = []

    while len(candidates) > 0:
        current = candidates.pop(0)
        group = [current]
        remaining = []
        
        # Suche vertikal gestapelte ODER überlappende Segmente
        for other in candidates:
            # 1. Distanz der Zentren (wie bisher, aber leicht erhöhter Threshold)
            dist_x = abs(other['center_x'] - current['center_x'])
            
            # 2. Horizontale Überlappung berechnen
            # Wie viele Pixel überschneiden sich auf der X-Achse?
            overlap_x = max(0, min(current['x2'], other['x2']) - max(current['x1'], other['x1']))
            
            # Verhältnis der Überlappung zur schmaleren der beiden Boxen
            min_width = min(current['width'], other['width'])
            overlap_ratio = overlap_x / min_width if min_width > 0 else 0
            
            # Wenn sie sehr nah beieinander liegen ODER sich stark überlappen -> Gruppe!
            if dist_x < x_thresh or overlap_ratio > overlap_thresh:
                group.append(other)
            else:
                remaining.append(other)
        
        candidates = remaining
        
        # Nimm das unterste Segment (größtes Y -> Boden)
        lowest = max(group, key=lambda c: c['y_bottom'])
        keep_indices.append(lowest['id'])

    return keep_indices

def rotate_point_back(u_rot, v_rot, w_rot, h_rot):
    """
    Rotiert einen Punkt zurück (von 90° CW -> Original).
    u_rot, v_rot: Koordinaten im rotierten Bild
    w_rot, h_rot: Maße des ROTIERTEN Bildes
    """
    # Formel für 90° CW Rückgängig:
    # Original X (u) = Rotiertes Y (v)
    # Original Y (v) = Rotierte Breite - 1 - Rotiertes X (u)
    
    u_orig = v_rot
    v_orig = w_rot - 1 - u_rot
    
    return int(u_orig), int(v_orig)

def main():
    # 1. Reader starten
    print("Initialisiere DataReader...")
    reader = DataReader(
        path_data_dir=DATA_DIR_PC, # Pfad wenn nötig anpassen
        rectify_images=True,
        stereo_images=False,
        load_dvso=True,               # WICHTIG: Lädt Tiefe & Trajektorie
        path_camera_calib_left=PATH_CALIB_PC, # Expliziter Pfad zur Kalibrierung, wenn nötig anpassen
        load_gps=False,
        load_lidar=False,
        load_object_detection=False
    )
    
    # 2. Modell laden
    print(f"Lade Modell von {MODEL_PATH}...")
    model = YOLO(MODEL_PATH_PC) # Hier Pfad wenn nötig anpassen
    print(f"Erkannte Klassen im Modell: {model.names}") # Zur Kontrolle im Terminal

    # 3. CSV Datei öffnen
    print(f"Erstelle Ergebnis-Datei: {CSV_FILENAME}")
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header schreiben
        writer.writerow(["Image_ID", "X_Global", "Y_Global", "Z_Tiefe_lokal"])

        image_counter = 0 # Zähler für Skip-Funktion

        #print("Starte Verarbeitung... (Drücke 'q' im Bildfenster zum Beenden)")  # Zum manuellen überprüfen ob das modell funktionier

        for data in reader:
            # Nur vollständige Bild-Daten verarbeiten
            if not isinstance(data, ImageData): continue

            # --- 1. JEDES 15. BILD LOGIK ---
            image_counter += 1
            if image_counter % SKIP_IMAGES != 0:
                continue


            if data.image is None: continue
            
            # Prüfen ob DVSO Daten (Tiefe/Pose) da sind
            if data.dvso_data is None or data.dvso_data.trajectory is None:
                continue

            # --- VORBEREITUNG ---
            image = data.image
            
            # Rotation anwenden
            if ROTATE_IMAGE:
                process_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            else:
                process_image = image

            # Kopie zum Malen
            annotated_frame = process_image.copy()

            # --- A. YOLO INFERENCE ---
            results = model(process_image, conf=CONFIDENCE, verbose=False)
            
            # Matrizen laden
            T_world_cam = data.dvso_data.trajectory 
            depth_map = data.dvso_data.image_depth 
            P = data.camera_calibration.camera_intrinsic
            fx, fy = P[0, 0], P[1, 1]
            cx, cy = P[0, 2], P[1, 2]

            # --- B. VERARBEITUNG DER OBJEKTE ---
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                
                # 1. Segmente filtern (nur unterste Teile)
                valid_indices = filter_vertical_segments(results[0].boxes, X_THRESHOLD)
                
                for idx in valid_indices:
                    # 2. Klassen-Filter (Baum vs. Pfosten)
                    cls_id = int(results[0].boxes[idx].cls[0].item())
                    class_name = model.names[cls_id]
                    
                    if class_name != TARGET_CLASS:
                        continue
                    
                    # 3. Kontur & Ankerpunkt holen
                    if results[0].masks is None: continue
                    contours = results[0].masks.xy[idx]
                    if len(contours) == 0: continue
                    
                    # Ankerpunkt im ROTIERTEN Bild (u, v)
                    bottom_point = contours[np.argmax(contours[:, 1])]
                    u_curr, v_curr = int(bottom_point[0]), int(bottom_point[1])
                    
                    # Zeichnen (Box & Punkt)
                    box = results[0].boxes[idx].xyxy[0].cpu().numpy().astype(int)
                    #cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) # Nur für manuelle Überprüfung des Models notwendig
                    #cv2.circle(annotated_frame, (u_curr, v_curr), 5, (0, 0, 255), -1) # Nur für manuelle Überprüfung des Models notwendig

                    # --- C. KOORDINATEN RÜCK-RECHNEN ---
                    if ROTATE_IMAGE:
                        h_rot, w_rot = process_image.shape[:2]
                        # u_orig, v_orig beziehen sich auf das ORIGINAL (unrotierte) Bild
                        u_orig, v_orig = rotate_point_back(u_curr, v_curr, w_rot, h_rot)
                    else:
                        u_orig, v_orig = u_curr, v_curr

                    # --- D. SKALIERUNG FÜR TIEFENKARTE ---
                    # Da Zmap kleiner als orignalbild 
                    h_img, w_img = image.shape[:2]
                    h_map, w_map = depth_map.shape[:2]
                    
                    scale_x = w_map / w_img
                    scale_y = h_map / h_img
                    
                    # Koordinaten auf Map-Größe schrumpfen
                    u_map = int(u_orig * scale_x)
                    v_map = int(v_orig * scale_y)

                    # --- E. TIEFE HOLEN ---
                    if not (0 <= v_map < h_map and 0 <= u_map < w_map):
                        continue
                    
                    # --- NEU: 3x3 Median Filter (Robuster gegen NaN-Löcher) ---
                    window = depth_map[max(0, v_map-1):v_map+2, max(0, u_map-1):u_map+2]
                    z_depth = np.nanmedian(window)
                    
                    # Validierung der Tiefe
                    if z_depth <= 0.1 or np.isnan(z_depth) or np.isinf(z_depth): 
                        continue
                    if z_depth > MAX_DEPTH:
                        continue

                    # --- F. 3D BERECHNUNG ---
                    # u_orig/v_orig (großes Bild) mit fx/fy (große Kalibrierung)
                    x_cam = (u_orig - cx) * z_depth / fx
                    y_cam = (v_orig - cy) * z_depth / fy
                    vec_hom = np.array([x_cam, y_cam, z_depth, 1.0])

                    gps_to_cam_left = np.array([
                            [0, 0, -1, 0],
                            [0, -1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]]).astype(np.float32)
                    cam_left_to_gps = np.linalg.inv(gps_to_cam_left)
                    
                    # Transformation in Welt-Koordinaten (Matrix @ Vektor)
                    pos_world = T_world_cam  @ gps_to_cam_left @ vec_hom #cam_left_to_gps
                    X_world, Y_world, Z_world = pos_world[0], pos_world[1], pos_world[2]
                    gps_coords = meter_to_gps_coords_old(X_world, Y_world, Z_world)
                    # --- G. SPEICHERN & ANZEIGE ---
                    #print(f"Bild {data.image_id}: {class_name} bei X={gps_coords[0]:.2f}m, Y={gps_coords[1]:.2f}m (Z={gps_coords[2]:.1f}m)")
                    
                    # CSV schreiben
                    writer.writerow([data.image_id, f"{gps_coords[0]:.8f}", f"{gps_coords[1]:.8f}", f"{z_depth:.2f}"])
                    
                    # Erzwingt sofortiges Speichern der Zeile auf Festplatte
                    f.flush()
                    
                    # Text ins Bild malen
                    #label_text = f"X:{gps_coords[0]:.1f} Y:{gps_coords[1]:.1f}" # Nur für manuelle Überprüfung des Models notwendig
                    #cv2.putText(annotated_frame, label_text, (box[0], box[1]-10), # Nur für manuelle Überprüfung des Models notwendig
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Nur für manuelle Überprüfung des Models notwendig

            # --- H. BILD ANZEIGEN ---
            # Zeigt das Bild an und wartet UNENDLICH lange auf Tastendruck
            #cv2.imshow("Tree Pipeline", annotated_frame) # Nur für manuelle Überprüfung des Models notwendig
            #key = cv2.waitKey(0) # 0 = Warten bis Taste gedrückt wird # Nur für manuelle Überprüfung des Models notwendig
            
            #if key == ord('q'): # Nur für manuelle Überprüfung des Models notwendig
                #print("Abbruch durch Benutzer.") # Nur für manuelle Überprüfung des Models notwendig
                #break # Nur für manuelle Überprüfung des Models notwendig

    #cv2.destroyAllWindows()
    print(f"Fertig! Daten gespeichert in {CSV_FILENAME}")

if __name__ == "__main__":
    main()