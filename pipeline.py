import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData

### Settings ###
#DATA_DIR = "D:\Studium\BACHELOR ARBEIT\samson_data_reader"      # Ordner, wo die REC-Dateien liegen, HomePC
DATA_DIR = r"C:\Studium\BA_Projekt\DATEN"                                   # Ordner in dem die REC-Dateien liegen, ArbeitsLapTop
MODEL_PATH = r"C:\Studium\BA_Projekt\BachelorArbeit\YOLO_Modell\baum_thesis\versuch_3_higherRes_Aug_300Epochs\weights\best.pt" # Pfad zum YOLO-Modell
PATH_CALIB = r"C:\Studium\BA_Projekt\BachelorArbeit\SAMSON3_SAMSON4_stereo.yaml"
USE_EVERY = 10                                                  # Jedes wie vielte Bild genutzt werden soll
X_THRESHOLD = 30                                                # Pixel: Wie weit dürfen Segmente seitlich versetzt sein?
CONFIDENCE = 0.25                                               # YOLO Sicherheitsschwelle
ROTATE_IMAGE = True                                             # True, wenn Bilder um 90° gedreht werden müssen

counter = 0


def filter_vertical_segments(boxes, masks, x_thresh=30):
    """
    Filtert vertikal gestapelte Segmente (ein Baum, der durch Äste geteilt wurde).
    Gibt nur das UNTERSTE Segment zurück (das den Boden berührt).
    
    Returns: Liste von Indizes, die wir behalten wollen.
    """
    if boxes is None or len(boxes) == 0:
        return []

    # Wir bauen eine Liste von Objekten: {'id': 0, 'center_x': 123, 'y_bottom': 450}
    candidates = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # Bounding Box
        center_x = (x1 + x2) / 2
        candidates.append({'id': i, 'center_x': center_x, 'y_bottom': y2})

    # Sortiere nach X-Position (links nach rechts)
    candidates.sort(key=lambda c: c['center_x'])

    keep_indices = []
    
    while len(candidates) > 0:
        current = candidates.pop(0)
        
        # Suche alle, die vertikal dazu gehören (ähnliches X)
        group = [current]
        
        # Wir prüfen die nächsten in der Liste (da sie sortiert sind)
        # Wir müssen eine Kopie der Liste zum Iterieren nehmen oder Indizes managen
        # Einfacher: Wir iterieren und entfernen Treffer aus 'candidates'
        
        remaining = []
        for other in candidates:
            if abs(other['center_x'] - current['center_x']) < x_thresh:
                group.append(other) # Gehört zum selben Baum
            else:
                remaining.append(other) # Ist ein anderer Baum (weiter rechts)
        
        candidates = remaining
        
        # Aus der Gruppe (Segmente eines Baums) nehmen wir nur das UNTERSTE
        # Das ist das mit dem größten y_bottom Wert
        lowest_segment = max(group, key=lambda c: c['y_bottom'])
        keep_indices.append(lowest_segment['id'])

    return keep_indices

def rotate_point_back(u_rot, v_rot, w_rot, h_rot):
    """
    Korrigierte Rück-Rotation (90 Grad CW -> Original).
    u_rot, v_rot: Koordinaten im rotierten Bild (Hochkant)
    w_rot, h_rot: Maße des rotierten Bildes
    """
    # Formel für 90° CW Rückgängig (Inverse):
    # Original X = Rotiertes Y
    # Original Y = Rotierte_Breite - 1 - Rotiertes X
        
    u_orig = v_rot
    v_orig = w_rot - 1 - u_rot  # HIER WAR DER FEHLER (w_rot statt h_rot)
    
    return int(u_orig), int(v_orig)

def main():

    ### Data Reader initialisieren
    dr = DataReader(
        path_data_dir=DATA_DIR,
        rectify_images=True,
        path_camera_calib_left=PATH_CALIB,
        stereo_images=False,                                        # True oder False je nachdem ob man mit Stereo bildern arbeitet
        load_dvso=True,                                             # True weil Bilder mit den Odemtrie Daten usw. angereichert werden sollen
        load_gps=False,
        load_lidar=False,
        image_size=None
    )
    counter = 0
    ### Modell laden ###
    model = YOLO(MODEL_PATH)
    
    ### CSV Datei für Bäume ###
    csv_filename = "gefundene_baeume.csv"
    print(f"Speichere Ergebnisse in {csv_filename}...")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header schreiben (Spaltennamen)
        writer.writerow(["Image_ID", "Tree_ID", "X_Koordinate", "Y_Koordinate", "Tiefe_m"])
    
    print("Starte Pipeline mit Segemnt_Bereinigung...")
    ### Pipeline startet ###
    
    
    for data in dr:
        
        # Nur Bilder verarbeiten
        if not isinstance(data, ImageData): continue
        if data.image is None: continue
        
        # Check ob DVSO Daten da sind
        if data.dvso_data is None or data.dvso_data.trajectory is None:
            continue

        # Only use every Nth image
        if counter % USE_EVERY != 0:
            counter += 1
            continue
        

        image = data.image

        ### Rotieren für  YOLO Modell ###
        if ROTATE_IMAGE:
            process_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            process_image = image

        h_rot, w_rot = process_image.shape[:2] # Maße des rotierten Bildes
        results = model(process_image, verbose=False)

        # DEBUG 1: Hat YOLO überhaupt was gefunden?
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"Bild {data.image_id}: YOLO sieht nichts! (Confidence zu hoch?)")
        else:
            print(f"Bild {data.image_id}: YOLO hat {len(results[0].boxes)} Objekte gefunden.")

        # Kopie zum Malen
        annotated_frame = process_image.copy()
        # Daten vorbereiten
        T_world_cam = data.dvso_data.trajectory
        depth_map = data.dvso_data.image_depth
        P_matrix = data.camera_calibration.camera_intrinsic
        fx, fy = P_matrix[0, 0], P_matrix[1, 1]
        cx, cy = P_matrix[0, 2], P_matrix[1, 2]


        if results[0].boxes is not None and results[0].masks is not None:

            ### Segmente filtern, nur die unteren Teile des Stammes benötigt ###
            valid_indices = filter_vertical_segments(
                results[0].boxes,
                results[0].masks,
                x_thresh=X_THRESHOLD
                )
            
            # DEBUG 2: Wie viele bleiben nach dem Filter übrig?
            print(f" -> Nach Filter: {len(valid_indices)} Bäume übrig.")
            
            ### über die gefilterten Indizes gehen ###
            for idx in valid_indices:

                # 1. Hole die Klassen-ID der aktuellen Box
                cls_id = int(results[0].boxes[idx].cls[0].item())
                
                # 2. Hole den lesbaren Namen (z.B. 'Baum' oder 'Pfosten')
                class_name = model.names[cls_id]
                
                # 3. PRÜFUNG: Wenn es KEIN Baum ist, überspringen!
                # WICHTIG: Ersetze 'Baum' mit dem Namen aus Schritt 1 (z.B. 'tree', 'stamm' etc.)
                if class_name != 'trunk': 
                    # print(f"  -> Ignoriere {class_name}") # Optionales Debugging
                    continue

                # Maske holen
                # Achtung: xy liefert Koordination der Kontur
                contours = results[0].masks.xy[idx]
                if len(contours) == 0: continue

                # C. ANKERPUNKT IM BILD (u, v)
                bottom_point = contours[np.argmax(contours[:, 1])]
                u_curr, v_curr = int(bottom_point[0]), int(bottom_point[1])
                
                # Malen im aktuellen Frame (Box + Punkt)
                box = results[0].boxes[idx].xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (u_curr, v_curr), 5, (0, 0, 255), -1)

                # D. KOORDINATEN FÜR 3D VORBEREITEN
                if ROTATE_IMAGE:
                    h_rot, w_rot = process_image.shape[:2]
                    u_orig, v_orig = rotate_point_back(u_curr, v_curr, w_rot, h_rot)
                else:
                    u_orig, v_orig = u_curr, v_curr

                # E. TIEFE HOLEN (Bounds Check im Original-Bild-Format)
                if not (0 <= v_orig < depth_map.shape[0] and 0 <= u_orig < depth_map.shape[1]):
                    print(f"  [SKIP] Punkt ({u_orig}, {v_orig}) liegt außerhalb der Tiefenkarte (Größe: {depth_map.shape})!")
                    continue
                
                z_depth = depth_map[v_orig, u_orig]
                
                if z_depth <= 0.1 or np.isnan(z_depth): 
                    print(f"  [SKIP] Ungültige Tiefe an ({u_orig}, {v_orig}): Wert ist {z_depth}")
                    continue

                # Manchmal sind Tiefen extrem groß (z.B. Himmel = 655m)
                if z_depth > 20.0: # Zum Beispiel: Ignoriere alles weiter als 20m
                     print(f"  [SKIP] Baum zu weit weg: {z_depth:.1f}m")
                     continue

                # F. 3D BERECHNUNG
                # Pixel -> Kamera
                x_cam = (u_orig - cx) * z_depth / fx
                y_cam = (v_orig - cy) * z_depth / fy
                vec_hom = np.array([x_cam, y_cam, z_depth, 1.0])
                
                # Kamera -> Welt
                pos_world = T_world_cam @ vec_hom
                X, Y = pos_world[0], pos_world[1]

                # G. AUSGABE
                print(f"Bild {data.image_id}: Baum bei X={X:.2f}m, Y={Y:.2f}m")
                

                # HIER: In die Datei schreiben
                # Wir speichern auch die Bild-ID, damit du später weißt, welches Foto das war
                writer.writerow([data.image_id, idx, f"{X:.4f}", f"{Y:.4f}", f"{z_depth:.2f}"])
                
            
                # Text ins Bild
                cv2.putText(annotated_frame, f"{X:.1f}/{Y:.1f}", (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Ausgabe auf Konsole
                print(f"Bild {data.image_id}: Baum gefunden bei X={X_world:.2f}m, Y={Y_world:.2f}m")
                
                # TODO Speichern in Map Objekt

            # Anzeigen lassen
            cv2.imshow("Filtered Localization", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)

            if key == ord('q'):
                break

        cv2.destroyAllWindows()

        counter += 1

if __name__ == "__main__":
    main()