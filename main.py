import os
import cv2
import asyncio
import telegram
import time
from ultralytics import YOLO
from dotenv import load_dotenv
import numpy as np

load_dotenv()
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
ip_camera_url = os.getenv("IP_CAMERA_URL")

model = YOLO('best.pt').to('cuda')

notif_cooldown = 60

async def send_alert_async(message):
    bot = telegram.Bot(token=bot_token)
    async with bot:
        await bot.send_message(chat_id=chat_id, text=message)

def send_telegram_alert(message):
    print("sending alert!")
    asyncio.run(send_alert_async(message))

def main():
    cap = cv2.VideoCapture(ip_camera_url)

    person_in_room = False
    last_notif = 0
    frame_skip = 3
    frame_count = 0

    while True:
        success, frame = cap.read()
        
        if not success or frame is None:
            time.sleep(0.1)
            continue
        
        frame_count +=1
        if frame_count % frame_skip != 0:
            continue

        results = model.track(frame, persist=False, conf=0.50, verbose=False)
        
        person_detected = False

        for r in results:
            if len(r.boxes) > 0:
                person_detected = True
                break
        
        if person_detected and not person_in_room:
            current_time = time.time()
            if current_time - last_notif > notif_cooldown:
                send_telegram_alert("A person has entered the room.")
                last_notif = current_time
            
            person_in_room = True

        elif not person_detected and person_in_room:
            person_in_room = False

        annotated_frame = results[0].plot()
        cv2.imshow("Room", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, exiting.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()