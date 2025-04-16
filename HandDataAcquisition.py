import os
import time
from time import strftime, localtime
import cv2
import numpy as np
import mediapipe as mp

def FER_live_cam():
    current_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    folder_name = f"Data/Hand_{current_time}"
    os.makedirs(folder_name, exist_ok=True)

    # 初始化MediaPipe手部检测
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 创建黑色背景
        masked_image = np.zeros_like(frame)
        save_flag = False

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            save_flag = True
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # 手势区域处理
                hull = cv2.convexHull(np.array(points))
                padding = 20
                x, y, bw, bh = cv2.boundingRect(hull)
                x = max(0, x - padding)
                y = max(0, y - padding)
                bw = min(w - x, bw + 2 * padding)
                bh = min(h - y, bh + 2 * padding)
                rect = (x, y, bw, bh)

                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y:y + bh, x:x + bw] = cv2.GC_PR_FGD
                cv2.drawContours(mask, [hull], -1, cv2.GC_FGD, -1)

                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

                mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

                # 形态学优化
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.erode(mask, kernel, iterations=1)

                # 生成单手的掩膜图像
                current_masked = cv2.bitwise_and(frame, frame, mask=mask)
                masked_image = cv2.bitwise_or(masked_image, current_masked)

                # 查找手势边界并裁剪
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(coords)
                    if w_crop > 0 and h_crop > 0:
                        cropped = current_masked[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(f'{folder_name}/{timestamp}.png', cropped)

        # 显示实时画面
        cv2.imshow('Hand Segmentation', masked_image if save_flag else frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FER_live_cam()
