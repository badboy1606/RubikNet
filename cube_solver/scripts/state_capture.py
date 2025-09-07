import cv2
import numpy as np

class RubiksCubeScanner:
    def __init__(self, size=150):
        self.size = size
        self.face_names = ["front", "back", "top", "bottom", "right", "left"]
        self.face_images = [f"{name}.jpeg" for name in self.face_names]
        self.colors_array = np.empty(54, dtype=object)
        self.index = 0

    # Capturing all the faces with camera
    def capture_faces(self):
        cap = cv2.VideoCapture(0)
        i = 0
        while i < 6:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            height, width = frame.shape[:2]
            centerx, centery = width // 2, height // 2

            cv2.rectangle(frame, (centerx - self.size, centery - self.size),
                          (centerx + self.size, centery + self.size), (0, 255, 0), 2)

            cv2.putText(frame, f"Place: {self.face_names[i].upper()} | Press 'C' to Capture",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imshow("Rubik's Cube Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                face_filename = self.face_images[i]
                face_crop = frame[centery - self.size:centery + self.size,
                                  centerx - self.size:centerx + self.size]
                cv2.imwrite(face_filename, face_crop)
                print(f"Saved {face_filename}")
                i += 1

        cap.release()
        cv2.destroyAllWindows()

    # Tuning 
    def classify_color(self, h, s, v):
        if s < 50 and v > 160:
            return "w"
        elif 5 <= h <= 15 and s > 115:
            return "o"
        elif h < 5 or h > 170:
            return "r"
        elif 40 <= h <= 80 and s > 100:
            return "g"
        elif 95 <= h <= 110 and s > 100:
            return "b"
        elif 20 <= h <= 35 and s > 100 and v > 150:
            return "y"
        else:
            return "unknown"

    # Processing
    def process_faces(self):
        for face_num, img_path in enumerate(self.face_images):
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = hsv.shape[:2]
            tile_h = height // 3
            tile_w = width // 3

            print(f"\nProcessing face: {self.face_names[face_num]}")
            for row in range(3):
                for col in range(3):
                    y = row * tile_h
                    x = col * tile_w
                    tile = hsv[y + tile_h // 4: y + 3 * tile_h // 4,
                               x + tile_w // 4: x + 3 * tile_w // 4]
                    h, s, v = np.mean(tile.reshape(-1, 3), axis=0)
                    color = self.classify_color(h, s, v)

                    if color == "unknown":
                        print(f"\nUnknown color detected at index a[{self.index}]")
                        print(f"HSV = ({int(h)}, {int(s)}, {int(v)})")
                        cv2.imshow(f"Tile a[{self.index}]", img[y:y + tile_h, x:x + tile_w])
                        cv2.waitKey(1)
                        manual = input("Please type the correct color (e.g. w/g/b/y/o/r): ")
                        color = manual
                        cv2.destroyAllWindows()

                    self.colors_array[self.index] = color
                    self.index += 1

            # Show detected face for verification
            start = self.index - 9
            print(f"\nDetected colors for {self.face_names[face_num]} face:")
            for i in range(start, start + 9):
                print(f"a[{i}] = {self.colors_array[i]}", end="\t")
                if (i - start + 1) % 3 == 0:
                    print()

            self.verify_face(start)

    # Verification
    def verify_face(self, start):
        while True:
            verify = input("Do you want to correct any color on this face? (yes/no): ")
            if verify == "no":
                break
            elif verify == "yes":
                edit_index = int(input("Enter index to correct (e.g. 14): "))
                new_color = input("Enter correct color: ")
                self.colors_array[edit_index] = new_color
                print(f"Updated a[{edit_index}] to {new_color}")
            else:
                print("Please answer with yes or no.")

    # Main workflow 
    def scan(self):
        self.capture_faces()
        self.process_faces()
        print("\nFinal Verified Colors:")
        for i in range(54):
            print(f"a[{i}] = {self.colors_array[i]}")
        return self.colors_array.copy()