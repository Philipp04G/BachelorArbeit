import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# --- IMPORTS DER MODULE ---
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData

# --- KONFIGURATION (PFADE ANPASSEN!) ---
DATA_DIR = r"C:\Studium\BA_Projekt\DATEN"
MODEL_PATH = r"C:\Studium\BA_Projekt\BachelorArbeit\YOLO_Modell\baum_thesis\versuch_3_higherRes_Aug_300Epochs\weights\best.pt"
PATH_CALIB = r"C:\Studium\BA_Projekt\BachelorArbeit\SAMSON4_SAMSON3_stereo.yaml"

# --- EINSTELLUNGEN ---
CSV_FILENAME = "gefundene_baeume.csv"
TARGET_CLASS = 'trunk' # <--- HIER DEN GENAUEN NAMEN AUS DEM DATENSATZ EINTRAGEN!
X_THRESHOLD = 30       # Pixel: Wie weit dürfen Segmente seitlich versetzt sein?
CONFIDENCE = 0.25      # YOLO Sicherheitsschwelle
ROTATE_IMAGE = True    # True, wenn Bilder um 90° gedreht werden müssen
MAX_DEPTH = 20.0       # Bäume weiter weg als 20m ignorieren
SKIP_IMAGES = 10       # Nur jedes X-te Bild verarbeiten

def filter_vertical_segments(boxes, x_thresh=30):
    """
    Behält nur das unterste Segment eines Baumes (falls der Stamm zerteilt wurde).
    Gibt die Indizes der Boxen zurück, die wir behalten wollen.
    """
    if boxes is None or len(boxes) == 0:
        return []

    candidates = []
    for i, box in enumerate(boxes):
        b = box.xyxy[0].cpu().numpy()
        center_x = (b[0] + b[2]) / 2
        y_bottom = b[3]
        candidates.append({'id': i, 'center_x': center_x, 'y_bottom': y_bottom})

    # Sortieren von links nach rechts
    candidates.sort(key=lambda c: c['center_x'])
    keep_indices = []

    while len(candidates) > 0:
        current = candidates.pop(0)
        group = [current]
        remaining = []
        
        # Suche vertikal gestapelte Segmente
        for other in candidates:
            if abs(other['center_x'] - current['center_x']) < x_thresh:
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
        path_data_dir=DATA_DIR,
        rectify_images=True,
        stereo_images=False,
        load_dvso=True,               # WICHTIG: Lädt Tiefe & Trajektorie
        path_camera_calib_left=PATH_CALIB, # Expliziter Pfad zur Kalibrierung
        load_gps=False,
        load_lidar=False,
        load_object_detection=False
    )
    
    # 2. Modell laden
    print(f"Lade Modell von {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"Erkannte Klassen im Modell: {model.names}") # Zur Kontrolle im Terminal

    # 3. CSV Datei öffnen
    print(f"Erstelle Ergebnis-Datei: {CSV_FILENAME}")
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header schreiben
        writer.writerow(["Image_ID", "Tree_Index", "Class", "X_Global", "Y_Global", "Z_Tiefe_lokal"])

        image_counter = 0 # Zähler für Skip-Funktion

        print("Starte Verarbeitung... (Drücke 'q' im Bildfenster zum Beenden)")

        for data in reader:
            # Nur vollständige Bild-Daten verarbeiten
            if not isinstance(data, ImageData): continue

            # --- 1. JEDES 10. BILD LOGIK ---
            image_counter += 1
            if image_counter % SKIP_IMAGES != 0:
                # Überspringe dieses Bild, aber gib kurzes Feedback
                # print(f"Skippe Bild {image_counter}...") 
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
                        # print(f"  -> Ignoriere {class_name}")
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
                    cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (u_curr, v_curr), 5, (0, 0, 255), -1)

                    # --- C. KOORDINATEN RÜCK-RECHNEN ---
                    if ROTATE_IMAGE:
                        h_rot, w_rot = process_image.shape[:2]
                        # u_orig, v_orig beziehen sich auf das ORIGINAL (unrotierte) Bild
                        u_orig, v_orig = rotate_point_back(u_curr, v_curr, w_rot, h_rot)
                    else:
                        u_orig, v_orig = u_curr, v_curr

                    # --- D. SKALIERUNG FÜR TIEFENKARTE ---
                    # Das Originalbild ist z.B. 1000x1000, die Map aber nur 200x200 
                    # TODO !!!!! Vielleicht hier ein fehler??????
                    h_img, w_img = image.shape[:2]
                    h_map, w_map = depth_map.shape[:2]
                    
                    scale_x = w_map / w_img
                    scale_y = h_map / h_img
                    
                    # Koordinaten auf Map-Größe schrumpfen
                    u_map = int(u_orig * scale_x)
                    v_map = int(v_orig * scale_y)

                    # --- E. TIEFE HOLEN ---
                    if not (0 <= v_map < h_map and 0 <= u_map < w_map):
                        # print(f"  [SKIP] Außerhalb Map: ({u_map}, {v_map})")
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

                    gps_to_cam_left = np.array([[0, 0, -1, 0],
                            [0, -1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]]).astype(np.float32)
                    cam_left_to_gps = np.linalg.inv(gps_to_cam_left)
                    
                    # Transformation in Welt-Koordinaten (Matrix @ Vektor)
                    pos_world = T_world_cam @ vec_hom @ cam_left_to_gps
                    X_world, Y_world = pos_world[0], pos_world[1]

                    # --- G. SPEICHERN & ANZEIGE ---
                    print(f"Bild {data.image_id}: {class_name} bei X={X_world:.2f}m, Y={Y_world:.2f}m (Z={z_depth:.1f}m)")
                    
                    # CSV schreiben
                    writer.writerow([data.image_id, idx, class_name, f"{X_world:.4f}", f"{Y_world:.4f}", f"{z_depth:.2f}"])
                    
                    # Text ins Bild malen
                    label_text = f"X:{X_world:.1f} Y:{Y_world:.1f}"
                    cv2.putText(annotated_frame, label_text, (box[0], box[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- H. BILD ANZEIGEN ---
            # Zeigt das Bild an und wartet UNENDLICH lange auf Tastendruck
            cv2.imshow("Tree Pipeline", annotated_frame)
            key = cv2.waitKey(0) # 0 = Warten bis Taste gedrückt wird
            
            if key == ord('q'): 
                print("Abbruch durch Benutzer.")
                break

    cv2.destroyAllWindows()
    print(f"Fertig! Daten gespeichert in {CSV_FILENAME}")

if __name__ == "__main__":
    main()