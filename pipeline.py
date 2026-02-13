import os
import cv2
import numpy as np
from ultralytics import YOLO
from samson_data_reader.data_reader import DataReader
from samson_data_reader.datacalsses import ImageData

### Settings ###
#DATA_DIR = "D:\Studium\BACHELOR ARBEIT\samson_data_reader"      # Ordner, wo die REC-Dateien liegen, HomePC
DATA_DIR = "C:\Studium\DATEN"                                   # Ordner in dem die REC-Dateien liegen, ArbeitsLapTop
MODEL_PATH = "C:\Studium\YOLO_Modell\baum_thesis\versuch_3_higherRes_Aug_300Epochs\weights\best.pt" # Pfad zum YOLO-Modell
USE_EVERY = 10                                                  # Jedes wie vielte Bild genutzt werden soll
X_THRESHOLD = 30                                                # Pixel: Wie weit dürfen Segmente seitlich versetzt sein?



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



def main():

    ### Data Reader initialisieren
    dr = DataReader(
        path_data_dir=DATA_DIR,
        stereo_images=False,                                        # True oder False je nachdem ob man mit Stereo bildern arbeitet
        load_dvso=True,                                             # True weil Bilder mit den Odemtrie Daten usw. angereichert werden sollen
        load_gps=False,
        load_lidar=False,
        image_size=None
    )

    ### Modell laden ###
    model = YOLO(MODEL_PATH)

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
        annotated_frame = image.copy()
        image_rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h_rot, w_rot = image_rot.shape[:2] # Maße des rotierten Bildes
        results = model(image_rot, verbose=False)

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
            
            ### über die gefilterten Indizes gehen ###
            for idx in valid_indices:

                # Maske holen
                # Achtung: xy liefert Koordination der Kontur
                contours = results[0].masks.xy[idx]
                if len(contours) == 0: continue

                # Punkt unten Mitte
                bottom_point = contours[np.argmax(contours[:, 1])]
                u, v = int(bottom_point[0]), int(bottom_point[1])

                # Tiefe prüfen, bounds check
                if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
                    continue
                z_depth = depth_map[v, u]

                # auf ungültige Tiefe prüfen
                if z_depth <= 0.1 or np.isnan(z_depth):
                    pass

                if z_depth <= 0.1 or np.isnan(z_depth): continue

                # Pixel -> Kamera-Koordinaten
                x_cam = (u - cx) * z_depth / fx
                y_cam = (v - cy) * z_depth / fy

                vec_cam_hom = np.array([x_cam, y_cam, z_depth, 1.0])

                # Kamera -> Weltkoordinaten
                pos_world_hom = T_world_cam @ vec_cam_hom
                X_world, Y_world = pos_world_hom[0], pos_world_hom[1]

                # Visualisierung zur Überprüfung
                box = results[0].boxes[idx].xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Punkt und Koordinaten zeichnen
                cv2.circle(annotated_frame, (u, v), 6, (0, 0, 255), -1)
                text = f"X:{X_world:.1f} Y:{Y_world:.1f}"
                cv2.putText(annotated_frame, text, (u, v-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # TODO Speichern in Map Objekt


        print(f"Exported ")
        img_counter += 1
        counter += 1

    print("\n------------------------------------------------")
    print("✅ Export vollständig abgeschlossen!")
    print("→ Train Bilder:", train_counter)
    print("→ Val Bilder:  ", val_counter)
    print("→ Ausgabeordner:", OUT_DIR)
    print("------------------------------------------------")