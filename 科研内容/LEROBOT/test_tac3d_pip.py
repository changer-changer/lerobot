import sys
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.tactile.configs import Tac3DSensorConfig
from lerobot.tactile.direct_connection.tac3d_sensor import Tac3DTactileSensor

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing Tac3D pip package integration via LeRobot Sensor Abstraction...")
    
    # Defaults to UDP port 9988
    config = Tac3DSensorConfig(udp_port=9988)
    
    print(f"Instantiating sensor with config: {config}")
    sensor = Tac3DTactileSensor(config)
    
    try:
        print("Connecting to sensor... (waiting for warmup frame)")
        sensor.connect(warmup=True)
        print("Sensor connected successfully.")
        
        print("Reading 30 frames of tactile data...")
        for i in range(30):
            data = sensor.read()
            # Depending on Tac3DSensorConfig.data_type, shape might be (400, 3) or (400, 6)
            print(f"Frame {i+1} | Extracted data shape: {data.shape}")
            time.sleep(1/30.0)
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Disconnecting sensor...")
        sensor.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()
